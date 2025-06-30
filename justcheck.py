import os
import warnings
import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, LLMChain
import jinja2
from dotenv import load_dotenv

load_dotenv()

# Disable warnings
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration
DB_ROOT = "vector_databases"
DATA_DB_DIR = os.path.join(DB_ROOT, "data")
CASES_DB_DIR = os.path.join(DB_ROOT, "case_laws")
TEMPLATES_DIR = "templates"

# Create directories if they don't exist
os.makedirs(DATA_DB_DIR, exist_ok=True)
os.makedirs(CASES_DB_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Initialize FastAPI with CORS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jinja2 templates
if os.path.exists(TEMPLATES_DIR):
    template_loader = jinja2.FileSystemLoader(searchpath=TEMPLATES_DIR)
    template_env = jinja2.Environment(loader=template_loader)
else:
    template_env = None

# Pydantic models
class DocumentSource(BaseModel):
    document_id: str
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class QuestionRequest(BaseModel):
    question: str
    collection: str
    k: int = 3

class AnswerResponse(BaseModel):
    answer: str
    sources: List[DocumentSource]
    collection: str
    prompt_type: str

class SearchRequest(BaseModel):
    question: str
    collections: List[str]
    k: int = 3

class SearchStatusResponse(BaseModel):
    search_id: str
    status: str
    completed_collections: List[str]
    results: Dict[str, Optional[AnswerResponse]]
    error: Optional[str] = None

# Global state for active searches
active_searches: Dict[str, Dict] = {}

def initialize_vectorstore(collection_name: str, persist_dir: str) -> Optional[Chroma]:
    """Initialize a ChromaDB vector store if it exists."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        
        if os.path.exists(persist_dir):
            print(f"\nLoading existing vectorstore from {persist_dir}")
            return Chroma(
                persist_directory=persist_dir,
                collection_name=f"ibc_{collection_name}",
                embedding_function=embeddings
            )
        return None
    except Exception as e:
        print(f"Error initializing vectorstore {collection_name}: {str(e)}")
        return None

# Initialize collections at startup
print("Starting server initialization...")
start_time = time.time()
collections = {}

# Expected collections and their paths
expected_collections = {
    "acts": os.path.join(DATA_DB_DIR, "acts"),
    "rules": os.path.join(DATA_DB_DIR, "rules"),
    "supreme_court": os.path.join(CASES_DB_DIR, "supreme_court"),
    "high_court": os.path.join(CASES_DB_DIR, "high_court"),
    "nclat": os.path.join(CASES_DB_DIR, "nclat"),
    "nclt": os.path.join(CASES_DB_DIR, "nclt")
}

try:
    # Load existing vectorstores
    for collection_name, persist_dir in expected_collections.items():
        vectorstore = initialize_vectorstore(collection_name, persist_dir)
        if vectorstore:
            collections[collection_name] = vectorstore
            print(f"Loaded collection: {collection_name}")
        else:
            print(f"Vector database not found for collection: {collection_name}")
    
    if not collections:
        raise RuntimeError("No valid vector databases found. Please initialize the databases first.")
    
    print(f"\nInitialized collections: {list(collections.keys())}")
    elapsed = time.time() - start_time
    print(f"Server initialization completed in {elapsed:.1f} seconds")
except Exception as e:
    print(f"\nInitialization failed: {str(e)}")
    raise

# Prompt Templates (replace with your actual templates)
# ... (keep all the imports and initial setup the same until the PROMPT_TEMPLATES section)

# Improved Prompt Templates with better summarization instructions
PROMPT_TEMPLATES = {
    "acts": """Analyze the following legal context carefully and provide a concise yet comprehensive answer to the question.
Focus on the most relevant sections and key provisions. If the answer isn't in the context, say so.

Relevant Context:
{context}

Question: {question}

Guidelines for your answer:
1. Begin with a direct answer if possible
2. Cite specific sections/references
3. Keep it professional and precise
4. If unsure, say "The context doesn't provide enough information"

Answer:""",
    
    "rules": """Extract key information from these rules to answer the question accurately.
Prioritize clarity and relevance over verbosity.

Relevant Rules:
{context}

Question: {question}

Answer Structure:
1. Direct response to question
2. Supporting rule references
3. Practical implications if relevant

Answer:""",
    
    "supreme_court": """Analyze this legal judgment carefully and provide a nuanced answer.
Focus on the ratio decidendi and key observations.

Case Context:
{context}

Question: {question}

Answer Guidelines:
1. Start with the legal principle established
2. Reference key paragraphs/judges' observations
3. Distinguish between obiter and ratio if relevant
4. Keep analysis focused on the question

Legal Analysis:""",
    
    "high_court": """Interpret this high court judgment to answer the question precisely.
Highlight the binding aspects and reasoning.

Judgment Excerpts:
{context}

Question: {question}

Response Structure:
1. Clear answer based on judgment
2. Paragraph references
3. Judicial reasoning summary
4. Limitations/scope if relevant

Analysis:""",
    
    "nclat": """Extract the core legal determination from this NCLAT order to answer the question.
Focus on the operative portions.

Tribunal Order:
{context}

Question: {question}

Answer Approach:
1. State the tribunal's finding
2. Reference order paragraphs
3. Explain legal basis briefly
4. Note any precedents cited

Determination:""",
    
    "nclt": """Analyze this NCLT order to provide a focused answer to the question.
Emphasize the dispositive reasoning.

Order Context:
{context}

Question: {question}

Response Format:
1. Direct answer from order
2. Key reasoning points
3. Relevant sections cited
4. Practical implications if any

Order Analysis:"""
}

# ... (keep all the remaining code the same)

def perform_search(search_id: str, question: str, collections_to_search: List[str], k: int):
    """Background task to perform the search across multiple collections"""
    try:
        for collection_name in collections_to_search:
            if collection_name not in collections:
                active_searches[search_id]["completed_collections"].append(collection_name)
                active_searches[search_id]["results"][collection_name] = None
                continue
            
            template = PROMPT_TEMPLATES.get(collection_name)
            if not template:
                active_searches[search_id]["completed_collections"].append(collection_name)
                active_searches[search_id]["results"][collection_name] = None
                continue
            
            try:
                llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)
                prompt = ChatPromptTemplate.from_template(template)
                retriever = collections[collection_name].as_retriever(
                    search_kwargs={'k': k}
                )
                
                qa_chain = RetrievalQA.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    input_key="query",
                    output_key="result"
                )
                
                result = qa_chain({"query": question})
                
                sources = [
                    DocumentSource(
                        document_id=str(i),
                        page_content=doc.page_content,
                        metadata=doc.metadata,
                        score=None
                    ) for i, doc in enumerate(result["source_documents"])
                ]
                
                active_searches[search_id]["results"][collection_name] = AnswerResponse(
                    answer=result["result"],
                    sources=sources,
                    collection=collection_name,
                    prompt_type=collection_name
                )
                
            except Exception as e:
                logger.error(f"Error processing collection {collection_name}: {str(e)}")
                active_searches[search_id]["results"][collection_name] = None
            
            active_searches[search_id]["completed_collections"].append(collection_name)
            
            # Small delay to allow polling to work better
            time.sleep(0.5)
            
    except Exception as e:
        active_searches[search_id]["error"] = str(e)
    finally:
        active_searches[search_id]["status"] = "complete"

@app.post("/api/search/start", response_model=SearchStatusResponse)
async def start_search(request: SearchRequest, background_tasks: BackgroundTasks):
    """Initiate a new search across multiple collections"""
    search_id = str(uuid.uuid4())
    active_searches[search_id] = {
        "status": "in_progress",
        "question": request.question,
        "collections": request.collections,
        "completed_collections": [],
        "results": {},
        "error": None,
        "start_time": time.time()
    }
    
    background_tasks.add_task(
        perform_search,
        search_id,
        request.question,
        request.collections,
        request.k
    )
    
    return {
        "search_id": search_id,
        "status": "started",
        "completed_collections": [],
        "results": {},
        "error": None
    }

@app.get("/api/search/status/{search_id}", response_model=SearchStatusResponse)
async def get_search_status(search_id: str):
    """Check the status of a search"""
    if search_id not in active_searches:
        raise HTTPException(status_code=404, detail="Search not found")
    
    search_data = active_searches[search_id]
    
    if search_data["error"]:
        raise HTTPException(status_code=500, detail=search_data["error"])
    
    return {
        "search_id": search_id,
        "status": search_data["status"],
        "completed_collections": search_data["completed_collections"],
        "results": search_data["results"],
        "error": search_data["error"]
    }

@app.delete("/api/search/{search_id}")
async def cleanup_search(search_id: str):
    """Clean up a completed search"""
    if search_id in active_searches:
        elapsed = time.time() - active_searches[search_id]["start_time"]
        logger.info(f"Cleaning up search {search_id} (duration: {elapsed:.2f}s)")
        del active_searches[search_id]
    return {"status": "deleted"}

@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(request: QuestionRequest):
    """Single collection answer endpoint (legacy)"""
    try:
        logger.info(f"\nProcessing question for {request.collection}: {request.question}")
        start_time = time.time()
        
        if request.collection not in collections:
            raise HTTPException(404, f"Collection {request.collection} not found")
        
        template = PROMPT_TEMPLATES.get(request.collection)
        if not template:
            raise HTTPException(400, f"No prompt template for collection {request.collection}")
        
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(template)
        retriever = collections[request.collection].as_retriever(
            search_kwargs={'k': request.k}
        )
        
        qa_chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            input_key="query",
            output_key="result"
        )
        
        result = qa_chain({"query": request.question})
        
        sources = [
            DocumentSource(
                document_id=str(i),
                page_content=doc.page_content,
                metadata=doc.metadata,
                score=None
            ) for i, doc in enumerate(result["source_documents"])
        ]
        
        elapsed = time.time() - start_time
        logger.info(f"Question processed in {elapsed:.1f} seconds")
        
        return AnswerResponse(
            answer=result["result"],
            sources=sources,
            collection=request.collection,
            prompt_type=request.collection
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(500, str(e))

@app.get("/api/collections")
async def list_collections():
    """List available collections"""
    return JSONResponse({
        "collections": list(collections.keys()),
        "status": "success"
    })

@app.post("/api/collections")
async def post_collections():
    """List available collections (POST version)"""
    return await list_collections()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "collections_loaded": len(collections),
        "active_searches": len(active_searches),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Root endpoint with HTML response"""
    if template_env:
        try:
            template = template_env.get_template("index.html")
            return HTMLResponse(content=template.render(collections=list(collections.keys())))
        except jinja2.TemplateNotFound:
            pass
    
    return HTMLResponse(content=f"""
        <html>
            <body>
                <h1>IBC Chatbot API</h1>
                <p>Available collections: {', '.join(collections.keys())}</p>
                <p>Visit <a href="/docs">/docs</a> for API documentation</p>
            </body>
        </html>
    """)

if __name__ == "__main__":
    import uvicorn
    print("\nStarting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)