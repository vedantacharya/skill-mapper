import openai
from PyPDF2 import PdfReader
from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from pymongo import MongoClient
from typing import List, Optional
import uuid
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from langchain_community.tools import DuckDuckGoSearchResults, YouTubeSearchTool
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import PromptTemplate
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="AI Matching System",
    description="AI-powered resume-JD matching platform for candidates and recruiters",
    version="1.1.0",
)

# ------------------- Configuration -------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "careerhub"
VECTOR_SIZE = 1536  # OpenAI embedding size

# Initialize Qdrant and MongoDB clients
qdrant = QdrantClient(url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY"))
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
users_collection = db["users"]
resume_collection = db["resume_metadata"]
jd_collection = db["jd_metadata"]

def ensure_qdrant_collection(collection_name: str):
    """
    Ensure that a given Qdrant collection exists. If not, create it.
    """
    collections = qdrant.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance="Cosine")
        )

# Ensure necessary collections exist
ensure_qdrant_collection("JD_Class")
ensure_qdrant_collection("Resume_Class")
# (Assume a "Knowledge_Base" collection exists for our knowledge base)

# Initialize OpenAI embedding model
embedding_model = OpenAIEmbedding()

# ------------------- Pydantic Models -------------------

class JDMetadata(BaseModel):
    jd_id: str
    recruiter_id: str
    Domain: str
    jd_role: str
    location: str
    jd_text: str
    jd_file_name: Optional[str] = None

class ResumeMetadata(BaseModel):
    resume_id: str
    candidate_id: str
    job_role: str
    resume_file_name: Optional[str] = None

# ------------------- Helper Functions -------------------

def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the given text using the OpenAI embedding model.
    """
    return embedding_model.get_text_embedding(text)

def extract_text_from_file(file: UploadFile) -> Optional[str]:
    """
    Extract text from an uploaded PDF or TXT file.
    
    Args:
        file: The uploaded file.
        
    Returns:
        Extracted text if successful; otherwise raises an HTTPException.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    try:
        file.file.seek(0)
        if file_ext == ".pdf":
            reader = PdfReader(file.file)
            texts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
            text = "\n".join(texts)
            return text.strip() if text.strip() else None
        elif file_ext == ".txt":
            content = file.file.read().decode("utf-8")
            return content.strip() if content.strip() else None
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and TXT allowed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

def get_user(email: str):
    """
    Retrieve a user from the MongoDB 'users' collection by email.
    """
    return users_collection.find_one({"email": email})

def parsing(text: str, model: str = "gpt-4o") -> str:
    """
    Parse the resume text using OpenAI's Chat API.
    
    The system prompt instructs the model to extract and structure details
    into a JSON object with keys:
      - 'name': candidate's full name
      - 'experience': total years and months of experience
      - 'skills': list of technical skills
      - 'project': list of projects and their details
      - 'education': educational background details
      - 'content': a brief summary of the overall resume content
    
    Args:
        text: The resume text to parse.
        model: Model identifier (default is "gpt-4o").
        
    Returns:
        A string containing the JSON output.
    """
    client = openai.OpenAI()  # Initialize the OpenAI client

    system_prompt = (
        "You are a resume parsing expert. Your task is to extract and structure the "
        "following details from the given resume text into a JSON object. The JSON "
        "should contain the following keys:\n"
        "  - 'name': candidate's full name\n"
        "  - 'experience': total years and months of experience\n"
        "  - 'skills': list of technical skills\n"
        "  - 'project': list of projects and their details\n"
        "  - 'education': educational background details\n"
        "  - 'content': a brief summary of the overall resume text content\n\n"
        "Do not include any extra commentary. Ensure the output is valid JSON."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# ------------------- Suggestion & Evaluation Tools -------------------

class SuggestionAndEvaluationTools:
    """
    Tools for the Suggestion and Evaluation agent.
    Provides methods to search the knowledge base, perform web searches, and search YouTube.
    """
    def __init__(self, index):
        # 'index' can be used if you have a document retrieval index.
        self.index = index

    def check_knowledge_base(self, query: str) -> str:
        """
        Search the knowledge base collection for relevant documents.
        
        Uses the Qdrant 'Knowledge_Base' collection.
        """
        embedding = get_embedding(query)
        results = qdrant.search(collection_name="Knowledge_Base", query_vector=embedding, limit=5)
        if not results:
            return "No matching documents found in the knowledge base."
        return "\n\n".join([
            f"Match {i+1} (Score: {result.score:.2f}):\n{result.payload.get('text', 'No text')}"
            for i, result in enumerate(results)
        ])

    @staticmethod
    def web_search(query: str) -> str:
        """
        Search the web using DuckDuckGo for related materials.
        """
        search = DuckDuckGoSearchResults()
        try:
            results = search.run(query)
            if isinstance(results, str):
                return results
            elif isinstance(results, list):
                return "\n\n".join([
                    f"{i+1}. {r.get('title', 'No Title')} - {r.get('url', 'No URL')}"
                    for i, r in enumerate(results[:5])
                ])
            return "Unexpected web search response format."
        except Exception as e:
            return f"Web search failed: {str(e)}"

    @staticmethod
    def youtube_search(query: str) -> str:
        """
        Search YouTube for study material videos.
        """
        try:
            youtube_tool = YouTubeSearchTool()
            results = youtube_tool.run(f"{query} study material")
            return results
        except Exception as e:
            return f"YouTube search failed: {str(e)}"

# ------------------- Lifespan & Agent Initialization -------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context to initialize the Suggestion and Evaluation agent.
    
    Sets up the following tools:
      - Check KnowledgeBase (searches the 'Knowledge_Base' collection)
      - Web Search (using DuckDuckGo)
      - YouTube Search (for study material)
    
    Also defines a prompt template for the agent.
    """
    # In practice, set app.state.index to your actual document retrieval index.
    app.state.index = None

    tools_instance = SuggestionAndEvaluationTools(app.state.index)
    app.state.tools = [
        Tool(
            name="Check KnowledgeBase",
            func=tools_instance.check_knowledge_base,
            description="Search the knowledge base for relevant documents."
        ),
        Tool(
            name="Web Search",
            func=tools_instance.web_search,
            description="Perform a web search using DuckDuckGo."
        ),
        Tool(
            name="YouTube Search",
            func=tools_instance.youtube_search,
            description="Search YouTube for study material videos."
        ),
    ]

    # Prompt template for the agent.
    template = (
        "You are an expert in Suggestion and Evaluation. "
        "You have access to the following tools: {tools}.\n\n"
        "When given a question, break down your reasoning into steps and use the tools as needed. "
        "After gathering information, provide a final answer starting with 'Final Answer:'.\n\n"
        "Question: {input}"
        "Thought: Explain your next action."
        "Action: Choose one of [{tool_names}]"
        "Action Input: Provide the input for that tool."
        "Observation: Record the tool's output."
        "... (Repeat as needed)"
        "Thought: I now have enough information."
        "Final Answer:"

        "Begin!"

        "Question: {input}\n"
        "Thought: {agent_scratchpad}"
    )
    prompt_template = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=template
    )
    
    app.state.agent_executor = AgentExecutor(
        agent=create_react_agent(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0.4),
            tools=app.state.tools,
            prompt=prompt_template
        ),
        tools=app.state.tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    yield

app.router.lifespan_context = lifespan

# ------------------- Recruiter Endpoints -------------------

@app.post("/recruiters/signup", response_model=dict, tags=["Recruiters"])
async def register_recruiter(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    company: str = Form(...),
    role: str = Form(...),
):
    """
    Register a new recruiter.
    """
    if get_user(email):
        raise HTTPException(400, "Email already registered")
    
    user_id = str(uuid.uuid4())
    user_data = {
        "_id": user_id,
        "name": name,
        "email": email,
        "password": password,  # Consider hashing passwords in production.
        "role": role,
        "company": company,
        "jds": []
    }
    users_collection.insert_one(user_data)
    return {"message": "Recruiter registered", "user_id": user_id}

# @app.post("/recruiters/jds", response_model=dict, tags=["Recruiters"])
# async def upload_jd(
#     recruiter_id: str = Form(...),
#     Domain: str = Form(...),
#     jd_role: str = Form(...),
#     location: str = Form(...),
#     jd_text: Optional[str] = Form(None),
#     jd_file: Optional[UploadFile] = File(None)
# ):
#     """
#     Upload a Job Description (JD).
#     """
#     recruiter = users_collection.find_one({"_id": recruiter_id, "role": "recruiter"})
#     if not recruiter:
#         raise HTTPException(403, "Forbidden: Recruiter not found")
    
#     if not jd_text and not jd_file:
#         raise HTTPException(400, "No JD provided")
    
#     text = jd_text if jd_text else extract_text_from_file(jd_file)
#     if not text:
#         raise HTTPException(400, "Failed to extract text from file")
    
#     file_name = jd_file.filename if jd_file else None
#     jd_id = str(uuid.uuid4())
#     embedding = get_embedding(text)
    
#     payload = {
#         "Domain": Domain,
#         "jd_role": jd_role,
#         "jd_text": text,
#         "recruiter_id": recruiter_id,
#         "jd_file_name": file_name
#     }
#     qdrant.upsert(
#         collection_name="JD_Class",
#         points=[models.PointStruct(id=jd_id, vector=embedding, payload=payload)]
#     )
    
#     jd_metadata = JDMetadata(
#         jd_id=jd_id,
#         recruiter_id=recruiter_id,
#         Domain=Domain,
#         jd_role=jd_role,
#         location =location,
#         jd_text=text,
#         jd_file_name=file_name
#     )
#     jd_collection.insert_one(jd_metadata.dict())
    
#     users_collection.update_one(
#         {"_id": recruiter_id},
#         {"$push": {"jds": {"jd_id": jd_id, "Domain": Domain, "role": jd_role}}}
#     )
    
#     return {"message": "JD uploaded", "jd_id": jd_id}

@app.post("/recruiters/jds", response_model=dict, tags=["Recruiters"])
async def upload_jd(
    recruiter_id: str = Form(...),
    Domain: str = Form(...),
    jd_role: str = Form(...),
    location: str = Form(...),
    jd_text: Optional[str] = Form(None),
    jd_file: Optional[UploadFile] = File(None)
):
    """
    Upload a Job Description (JD).
    """

    # Step 1: Validate recruiter
    recruiter = users_collection.find_one({"_id": recruiter_id, "role": "recruiter"})
    if not recruiter:
        
        raise HTTPException(status_code=403, detail="Forbidden: Recruiter not found")
    
    # Step 2: Validate JD input
    if not jd_text and not jd_file:
        
        raise HTTPException(status_code=400, detail="No JD provided")

    # Step 3: Extract JD text
    try:
        text = jd_text if jd_text else extract_text_from_file(jd_file)
        if not text:
            raise ValueError("Extracted text is empty")
    except Exception as e:
        
        raise HTTPException(status_code=400, detail=f"Failed to extract text from file: {str(e)}")

    # Step 4: Generate embedding
    try:
        embedding = get_embedding(text)
    except Exception as e:
        
        raise HTTPException(status_code=500, detail="Error generating embedding")

    # Step 5: Store JD in Qdrant
    jd_id = str(uuid.uuid4())
    file_name = jd_file.filename if jd_file else None
    payload = {
        "Domain": Domain,
        "jd_role": jd_role,
        "jd_text": text,
        "recruiter_id": recruiter_id,
        "jd_file_name": file_name
    }

    try:
        qdrant.upsert(
            collection_name="JD_Class",
            points=[models.PointStruct(id=jd_id, vector=embedding, payload=payload)]
        )
    except Exception as e:
       
        raise HTTPException(status_code=500, detail="Error inserting JD into Qdrant")

    # Step 6: Save JD metadata in MongoDB
    jd_metadata = JDMetadata(
        jd_id=jd_id,
        recruiter_id=recruiter_id,
        Domain=Domain,
        jd_role=jd_role,
        location=location,
        jd_text=text,
        jd_file_name=file_name
    )

    try:
        jd_collection.insert_one(jd_metadata.dict())
    except Exception as e:
        
        raise HTTPException(status_code=500, detail="Error inserting JD into database")

    # Step 7: Update recruiter profile in MongoDB
    try:
        update_result = users_collection.update_one(
            {"_id": recruiter_id},
            {"$push": {"jds": {"jd_id": jd_id, "Domain": Domain, "role": jd_role}}}
        )
        if update_result.modified_count == 0:
            raise ValueError("Failed to update recruiter profile")
    except Exception as e:
        
        raise HTTPException(status_code=500, detail="Error updating recruiter profile")

    
    return {"message": "JD uploaded", "jd_id": jd_id}


# ------------------- Candidate Endpoints -------------------

@app.post("/candidate/signup", response_model=dict, tags=["Candidate"])
async def register_candidate(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    designation: str = Form(...),
    role: str = Form(...),
    resume: UploadFile = File(...),
):
    """
    Register a new candidate and upload their resume.
    """
    if users_collection.find_one({"email": email}):
        raise HTTPException(400, "Email already registered")
    
    candidate_id = str(uuid.uuid4())

    resume_text = extract_text_from_file(resume)
    if not resume_text:
        raise HTTPException(400, "Failed to extract text from resume")
    
    # Parse the resume text to obtain structured details.
    parsed_resume = parsing(resume_text, model="gpt-4o")
    try:
        parsed_resume_json = json.loads(parsed_resume)
    except Exception as e:
        parsed_resume_json = {"error": "Json Load Fail", "raw_output": parsed_resume}
    
    qdrant.upsert(
        collection_name="Resume_Class",
        points=[models.PointStruct(
            id=candidate_id,
            vector=get_embedding(parsed_resume),
            payload={
                "name":name,
                "email": email,
                "designation": designation,
                "parsed_resume": parsed_resume_json
            }
        )]
    )
    
    users_collection.insert_one({
        "_id": candidate_id,
        "name": name,
        "email": email,
        "password": password,  # Consider securing passwords in production.
        "designation": designation,
        "role": role,
    })
    
    resume_collection.insert_one({
        "candidate_id": candidate_id,
        "content": resume_text,
        "designation": designation,
        "file_name": resume.filename,
        "parsed_resume": parsed_resume_json
    })
    
    return {"candidate_id": candidate_id, "parsed_resume": parsed_resume_json}

@app.put("/candidates/{candidate_id}", response_model=dict, tags=["Candidate Update"])
async def update_candidate(
    candidate_id: str,
    new_resume: Optional[UploadFile] = File(None),
    email: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    designation: Optional[str] = Form(None)
):
    """
    Update a candidate's profile with optional changes.
    """
    candidate = users_collection.find_one({"_id": candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    updates = {}
    
    if new_resume:
        resume_text = extract_text_from_file(new_resume)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from resume")
        
        parsed_resume = parsing(resume_text, model="gpt-4o")
        try:
            parsed_resume_json = json.loads(parsed_resume)
        except Exception as e:
            parsed_resume_json = {"error": "Parsing failed", "raw_output": parsed_resume}
        
        updates["vector"] = get_embedding(resume_text)
        updates["payload.resume_text"] = resume_text
        updates["payload.parsed_resume"] = parsed_resume_json
        
        resume_collection.update_one(
            {"candidate_id": candidate_id},
            {"$set": {"content": resume_text, "file_name": new_resume.filename, "parsed_resume": parsed_resume_json}}
        )
    
    if email:
        if users_collection.find_one({"email": email, "_id": {"$ne": candidate_id}}):
            raise HTTPException(status_code=400, detail="Email already in use")
        updates["payload.email"] = email
        users_collection.update_one({"_id": candidate_id}, {"$set": {"email": email}})
    
    if password:
        users_collection.update_one({"_id": candidate_id}, {"$set": {"password": password}})
    
    if designation:
        updates["payload.designation"] = designation
        users_collection.update_one({"_id": candidate_id}, {"$set": {"designation": designation}})
    
    if updates:
        qdrant_payload = {}
        if "payload.resume_text" in updates:
            qdrant_payload["resume_text"] = updates["payload.resume_text"]
        if "payload.parsed_resume" in updates:
            qdrant_payload["parsed_resume"] = updates["payload.parsed_resume"]
        if "payload.email" in updates:
            qdrant_payload["email"] = updates["payload.email"]
        if "payload.designation" in updates:
            qdrant_payload["designation"] = updates["payload.designation"]
        
        if qdrant_payload:
            qdrant.set_payload(
                collection_name="Resume_Class",
                points=[candidate_id],
                payload=qdrant_payload
            )
        
        if "vector" in updates:
            qdrant.upsert(
                collection_name="Resume_Class",
                points=[models.PointStruct(
                    id=candidate_id,
                    vector=updates["vector"],
                    payload=qdrant_payload
                )]
            )
    
    return {"message": "Profile updated successfully"}

# ------------------- Matching Endpoints -------------------

@app.get("/find_candidates/{jd_id}/matches", response_model=List[dict], tags=["Retrieving"])
async def get_candidate_matches(jd_id: str):
    """
    Retrieve candidate matches for a given JD.
    """
    jd_meta = jd_collection.find_one({"jd_id": jd_id})
    if not jd_meta:
        raise HTTPException(404, "JD not found")
    
    jd_embedding = get_embedding(jd_meta.get("jd_text", ""))
    matches = qdrant.search(
        collection_name="Resume_Class",
        query_vector=jd_embedding,
        limit=5
    )
    
    return [{"resume_id": match.id, "score": match.score} for match in matches]

@app.get("/find_job/{candidate_id}/matches", response_model=List[dict], tags=["Retrieving"])
async def get_recruiter_matches(candidate_id: str):
    """
    Retrieve recruiter matches for a given candidate.
    """
    resume_meta = resume_collection.find_one({"candidate_id": candidate_id})
    if not resume_meta:
        raise HTTPException(404, "Resume not found")
    
    candidate_embedding = get_embedding(resume_meta.get("content", ""))
    matches = qdrant.search(
        collection_name="JD_Class",
        query_vector=candidate_embedding,
        limit=2
    )
    
    return [{"jd_id": match.id, "score": match.score} for match in matches]

# ------------------- Suggestion and Evaluation Endpoints -------------------

@app.post("/candidate/suggestion", tags=["Suggestion"])
async def candidate_suggestion(candidate_id: str = Form(...)):
    """
    Candidate Suggestion API:
    
    Retrieves the top 5 JDs related to a candidate's resume and runs an agent to provide
    detailed suggestions—identifying skill gaps, recommending improvement areas, and returning
    study resource links (from web, knowledge base, and YouTube).
    
    Args:
        candidate_id: Candidate's unique ID.
        
    Returns:
        JSON with the candidate ID and suggestion analysis.
    """
    candidate = resume_collection.find_one({"candidate_id": candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    resume_text = candidate.get("content", "")
    designation = candidate.get("designation", "")
    candidate_embedding = get_embedding(resume_text + designation)
    jd_matches = qdrant.search(
        collection_name="JD_Class",
        query_vector=candidate_embedding,
        limit=5
    )
    if not jd_matches:
        raise HTTPException(status_code=404, detail="No JD matches found for candidate")
    
    jd_details = "\n\n".join([
        f"JD ID: {match.id}\nScore: {match.score:.2f}\nText: {match.payload.get('jd_text', 'No text')}"
        for match in jd_matches
    ])
    parsed_resume = candidate.get("parsed_resume", {})
    question = (
        f"Candidate Resume Analysis:\nParsed Resume: {json.dumps(parsed_resume)}\n\n"
        f"Top 5 matching Job Descriptions:\n{jd_details}\n\n"
        "Based on the above information and for given {designation}, identify the candidate's skill gaps, suggest improvement areas, "
        "and provide links to study materials from the web, knowledge base, and YouTube."
    )
    
    # Pass a dictionary with all required keys to the agent
    agent_response = app.state.agent_executor.run({
        "input": question,
        "agent_scratchpad": "",
        "tools": str(app.state.tools),
        "tool_names": ", ".join([tool.name for tool in app.state.tools])
    })
    return {"candidate_id": candidate_id, "suggestion_analysis": agent_response}


@app.post("/recruiter/evaluation", tags=["Evaluation"])
async def recruiter_evaluation(jd_id: str = Form(...)):
    """
    Recruiter Evaluation API:
    
    Retrieves the top 5 candidate resumes matching a given JD and runs an agent to provide an evaluation
    analysis—summarizing candidate strengths, weaknesses, and offering suggestions for improved candidate targeting.
    
    For evaluation, the agent is instructed to only use the necessary tools.
    
    Args:
        jd_id: Job Description unique ID.
        
    Returns:
        JSON with the JD ID and evaluation analysis.
    """
    jd = jd_collection.find_one({"jd_id": jd_id})
    if not jd:
        raise HTTPException(status_code=404, detail="JD not found")
    
    jd_text = jd.get("jd_text", "")
    jd_role = jd.get("jd_role", "")
    domain = jd.get("domain", "")
    jd_embedding = get_embedding(jd_text + jd_role + domain)
    candidate_matches = qdrant.search(
        collection_name="Resume_Class",
        query_vector=jd_embedding,
        limit=2
    )
    if not candidate_matches:
        raise HTTPException(status_code=404, detail="No candidate matches found")
    
    candidate_details = "\n\n".join([
        f"Candidate ID: {match.id}\nScore: {match.score:.2f}\nParsed Resume: {json.dumps(match.payload.get('parsed_resume', {}))}"
        for match in candidate_matches
    ])
    question = (
        f"Role:\n{jd_role}\n\n"
        f"Domain:\n{domain}\n\n"
        f"Job Description:\n{jd_text}\n\n"
        f"Top 5 Candidate Resumes:\n{candidate_details}\n\n"
        "Based on the above Role,Domain and Job Description, provide an evaluation analysis including candidate strengths, "
        "weaknesses, and suggestions to improve candidate targeting. Only use the necessary tools from your toolkit."
    )
    
    # For evaluation, if you wish to use fewer tools, you might modify the "tools" list here.
    # For example, only include Check KnowledgeBase and Web Search:
    evaluation_tools = [tool for tool in app.state.tools if tool.name in {"Check KnowledgeBase", "Web Search"}]
    
    agent_response = app.state.agent_executor.run({
        "input": question,
        "agent_scratchpad": "",
        "tools": str(evaluation_tools),
        "tool_names": ", ".join([tool.name for tool in evaluation_tools])
    })
    return {"jd_id": jd_id, "evaluation_analysis": agent_response}

# ------------------- Application Runner -------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
