import os
import warnings
import logging
import time
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
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
DATA_DIR = "Data"
ACTS_DIR = os.path.join(DATA_DIR, "Acts")
RULES_DIR = os.path.join(DATA_DIR, "Rules")
CASE_LAWS_DIR = "Case_Laws"

# Database directories
DB_ROOT = "vector_databases"
DATA_DB_DIR = os.path.join(DB_ROOT, "data")
CASES_DB_DIR = os.path.join(DB_ROOT, "case_laws")
TEMPLATES_DIR = "templates"

# Create database directories if they don't exist
os.makedirs(DATA_DB_DIR, exist_ok=True)
os.makedirs(CASES_DB_DIR, exist_ok=True)

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
template_loader = jinja2.FileSystemLoader(searchpath=TEMPLATES_DIR)
template_env = jinja2.Environment(loader=template_loader)

# Pydantic models
class DocumentSource(BaseModel):
    document_id: str
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class QuestionRequest(BaseModel):
    question: str
    collection: str  # e.g., "acts", "rules", "supreme_court", "high_court", etc.
    k: int = 3

class AnswerResponse(BaseModel):
    answer: str
    sources: List[DocumentSource]
    collection: str
    prompt_type: str

def print_progress(current, total, stage):
    """Helper function to print progress updates"""
    progress = (current / total) * 100
    print(f"\r[{stage}] Progress: {current}/{total} ({progress:.1f}%)", end="")
    if current == total:
        print()

def load_and_process_documents(directory: str) -> List[Any]:
    """Load and split documents from directory with progress tracking."""
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return []
        
    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=True
    )
    
    print(f"\nLoading documents from {directory}")
    documents = loader.load()
    
    if not documents:
        logger.warning(f"No documents loaded from {directory}")
        return []
        
    print("\nSplitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        separators=["\n\n", "\n•", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} document chunks")
    
    # Filter out empty documents
    valid_docs = [doc for doc in split_docs if doc.page_content.strip()]
    print(f"Keeping {len(valid_docs)} valid documents after filtering")
    
    return valid_docs

def initialize_or_load_vectorstore(collection_name: str, documents: List[Any], persist_dir: str) -> Chroma:
    """Initialize or load a ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    
    # Check if the vectorstore already exists
    if os.path.exists(persist_dir):
        print(f"\nLoading existing vectorstore from {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    
    if not documents:
        raise ValueError(f"No valid documents found for collection {collection_name}")
    
    print(f"\nCreating new vectorstore for {collection_name}")
    
    # Process in batches to show progress
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size
    vectorstore = None
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = (batch_num + 1) * batch_size
        batch_docs = documents[batch_start:batch_end]
        
        print_progress(batch_num + 1, total_batches, "PROCESSING")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_dir
            )
        else:
            vectorstore.add_documents(batch_docs)
    
    vectorstore.persist()
    return vectorstore

# Initialize collections at startup
print("Starting server initialization...")
start_time = time.time()
collections = {}

try:
    # Initialize Acts collection
    acts_db_path = os.path.join(DATA_DB_DIR, "acts")
    acts_docs = load_and_process_documents(ACTS_DIR)
    if acts_docs:
        collections["acts"] = initialize_or_load_vectorstore(
            "ibc_acts",
            acts_docs,
            acts_db_path
        )
    
    # Initialize Rules collection
    rules_db_path = os.path.join(DATA_DB_DIR, "rules")
    rules_docs = load_and_process_documents(RULES_DIR)
    if rules_docs:
        collections["rules"] = initialize_or_load_vectorstore(
            "ibc_rules",
            rules_docs,
            rules_db_path
        )
    
    # Initialize Case Laws collections for each court type
    for court in ["Supreme_Court", "High_Court", "NCLAT", "NCLT"]:
        court_dir = os.path.join(CASE_LAWS_DIR, court)
        court_db_path = os.path.join(CASES_DB_DIR, court.lower())
        court_docs = load_and_process_documents(court_dir)
        
        if court_docs:
            collection_name = f"ibc_{court.lower()}"
            collections[court.lower()] = initialize_or_load_vectorstore(
                collection_name,
                court_docs,
                court_db_path
            )
    
    if not collections:
        raise RuntimeError("No valid collections could be initialized")
    
    print(f"\nInitialized collections: {list(collections.keys())}")
    elapsed = time.time() - start_time
    print(f"Server initialization completed in {elapsed:.1f} seconds")
except Exception as e:
    print(f"\nInitialization failed: {str(e)}")
    raise

# Prompt Templates
ACTS_PROMPT_TEMPLATE = """
You are a legal expert specializing in the Insolvency and Bankruptcy Code (IBC) of India.
When answering questions about IBC Acts, you must:
1. Provide precise section citations (e.g., "Section 12 of IBC, 2016")
2. Include exact wording from the relevant sections when appropriate
3. Explain the legal implications clearly
4. Preserve any Hindi terms that appear in the context

Context: {context}

Question: {question}

Answer in clear language while preserving any Hindi terms:
"""

RULES_PROMPT_TEMPLATE = """
You are a legal expert specializing in the Insolvency and Bankruptcy Code (IBC) of India.
When answering questions about IBC Rules, you must:
1. Provide precise rule citations (e.g., "Rule 5 of IBC Rules, 2016")
2. Include exact wording from the relevant rules when appropriate
3. Explain the procedural implications clearly
4. Preserve any Hindi terms that appear in the context

Context: {context}

Question: {question}

Answer in clear language while preserving any Hindi terms:
"""

SUPREME_COURT_PROMPT_TEMPLATE = """
You are a legal analyst specializing in Indian Supreme Court case law related to insolvency.
When analyzing case law, you must:
1. Identify the case as a Supreme Court decision
2. Include full case citations (e.g., "ABC v. XYZ (2023) 5 SCC 123")
3. Highlight key legal principles established
4. Note the judicial reasoning
5. Preserve any Hindi terms that appear in the context

Context: {context}

Question: {question}

Analysis (preserve Hindi terms where present):
"""

HIGH_COURT_PROMPT_TEMPLATE = """
You are a legal analyst specializing in Indian High Court case law related to insolvency.
When analyzing case law, you must:
1. Identify the specific High Court (e.g., Delhi High Court)
2. Include full case citations with court and year
3. Highlight key legal principles established
4. Note the judicial reasoning
5. Preserve any Hindi terms that appear in the context

Context: {context}

Question: {question}

Analysis (preserve Hindi terms where present):
"""

NCLAT_PROMPT_TEMPLATE = """
You are a legal analyst specializing in NCLAT (National Company Law Appellate Tribunal) decisions.
When analyzing case law, you must:
1. Identify the case as an NCLAT decision
2. Include full case citations with year
3. Highlight key legal principles established
4. Note the judicial reasoning
5. Preserve any Hindi terms that appear in the context

Context: {context}

Question: {question}

Analysis (preserve Hindi terms where present):
"""

NCLT_PROMPT_TEMPLATE = """
You are a legal analyst specializing in NCLT (National Company Law Tribunal) decisions.
When analyzing case law, you must:
1. Identify the case as an NCLT decision
2. Include full case citations with bench and year
3. Highlight key legal principles established
4. Note the judicial reasoning
5. Preserve any Hindi terms that appear in the context

Context: {context}

Question: {question}

Analysis (preserve Hindi terms where present):
"""

# Map collections to their appropriate prompt templates
PROMPT_TEMPLATES = {
    "acts": ACTS_PROMPT_TEMPLATE,
    "rules": RULES_PROMPT_TEMPLATE,
    "supreme_court": SUPREME_COURT_PROMPT_TEMPLATE,
    "high_court": HIGH_COURT_PROMPT_TEMPLATE,
    "nclat": NCLAT_PROMPT_TEMPLATE,
    "nclt": NCLT_PROMPT_TEMPLATE
}

# API Endpoints
@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(request: QuestionRequest):
    try:
        print(f"\nProcessing question for {request.collection}: {request.question}")
        start_time = time.time()
        
        vectorstore = collections.get(request.collection)
        if not vectorstore:
            raise HTTPException(404, f"Collection {request.collection} not found")
        
        # Get appropriate prompt template
        template = PROMPT_TEMPLATES.get(request.collection)
        if not template:
            raise HTTPException(400, f"No prompt template defined for collection {request.collection}")
        
        # Create retrieval chain
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': request.k}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("Searching documents...")
        result = qa_chain({"query": request.question})
        
        # Process sources
        sources = []
        for i, doc in enumerate(result["source_documents"]):
            sources.append(DocumentSource(
                document_id=str(i),
                page_content=doc.page_content,
                metadata=doc.metadata,
                score=None  # Similarity score not available in this approach
            ))
        
        elapsed = time.time() - start_time
        print(f"Question processed in {elapsed:.1f} seconds")
        
        return AnswerResponse(
            answer=result["result"],
            sources=sources,
            collection=request.collection,
            prompt_type=request.collection
        )
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        raise HTTPException(500, str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Received: {data}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Basic frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    template = template_env.get_template("index.html")
    return HTMLResponse(content=template.render(collections=list(collections.keys())))

@app.get("/api/collections")
async def list_collections():
    return {
        "collections": list(collections.keys()),
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    print("\nStarting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)