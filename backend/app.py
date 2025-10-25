from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from backend.store_index import retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from backend.src.prompt import prompt
import os
import logging
from pathlib import Path
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Finance Bot API",
    description="An advanced RAG API with hybrid retrieval and reranking for finance questions.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get frontend directory
current_dir = Path(__file__).parent
frontend_dir = current_dir.parent / "frontend"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Initialize models
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,  # Lower temperature for more consistent responses
    max_tokens=1000
)

# Enhanced Models
class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the main types of financial statements?", description="User's finance-related query")
    use_reranking: bool = Field(default=True, description="Whether to use advanced reranking")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return")

class DocumentResponse(BaseModel):
    content: str
    source: str
    page: int
    relevance_score: float

class QueryResponse(BaseModel):
    status: str = Field(..., example="success")
    answer: str = Field(..., example="The main financial statements are income statement, balance sheet, and cash flow statement.")
    documents: List[DocumentResponse] = Field(default=[], description="Retrieved documents")
    retrieval_time: float = Field(..., description="Time taken for retrieval in seconds")
    error: Optional[str] = Field(default=None, example="None")

# Routes
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Welcome endpoint with available routes"""
    return {
        "message": "Welcome to Advanced Finance Bot API 🚀",
        "version": "2.0.0",
        "features": ["Hybrid Retrieval", "Reranking", "BM25 + Semantic Search"],
        "routes": {
            "health": "/health",
            "query": "/query",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Utility"])
async def health_check() -> Dict[str, str]:
    """Health check for uptime monitoring"""
    return {"status": "ok", "retriever": "active"}

@app.post("/query", response_model=QueryResponse, tags=["Query"], status_code=status.HTTP_200_OK)
async def query_docs(request: QueryRequest) -> QueryResponse:
    """Handles finance-related queries using advanced RAG pipeline"""
    start_time = time.time()
    
    try:
        # Retrieve relevant documents using advanced retriever
        retrieved_docs = retriever.retrieve(
            query=request.query,
            k=request.max_results,
            use_reranking=request.use_reranking
        )
        
        # Debug: Check if retrieved_docs contains proper Document objects
        if not retrieved_docs:
            logger.warning("No documents retrieved for query")
            return QueryResponse(
                status="error",
                answer="No relevant documents found for your query.",
                documents=[],
                retrieval_time=time.time() - start_time,
                error="No documents retrieved"
            )
        
        # Validate document objects
        for i, doc in enumerate(retrieved_docs):
            if not hasattr(doc, 'page_content'):
                logger.error(f"Document {i} is not a proper Document object: {type(doc)}")
                return QueryResponse(
                    status="error",
                    answer="Error processing retrieved documents.",
                    documents=[],
                    retrieval_time=time.time() - start_time,
                    error=f"Invalid document type at index {i}: {type(doc)}"
                )
        
        retrieval_time = time.time() - start_time
        
        # Prepare document responses
        document_responses = []
        for i, doc in enumerate(retrieved_docs):
            document_responses.append(DocumentResponse(
                content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", 0),
                relevance_score=1.0 - (i / len(retrieved_docs))  # Simple relevance scoring
            ))
        
        # Build QA chain with documents
        qa_chain = create_stuff_documents_chain(chat_model, prompt)
        
        # Generate answer using documents directly
        response = qa_chain.invoke({
            "context": retrieved_docs,
            "input": request.query
        })
        
        total_time = time.time() - start_time
        
        return QueryResponse(
            status="success",
            answer=response,
            documents=document_responses,
            retrieval_time=retrieval_time,
            error=None
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(
            status="error",
            answer="",
            documents=[],
            retrieval_time=time.time() - start_time,
            error=str(e)
        )

@app.get("/app")
async def serve_frontend():
    if frontend_dir.exists():
        return FileResponse(str(frontend_dir / "index.html"))
    else:
        return {"error": "Frontend not found"}

@app.get("/app.js")
async def serve_js():
    if frontend_dir.exists():
        return FileResponse(str(frontend_dir / "app.js"))
    else:
        return {"error": "Frontend JS not found"}
