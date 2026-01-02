from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import os
import glob
from uuid import uuid4
from datetime import datetime
from contextlib import asynccontextmanager
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────────────────────
# LANGGRAPH STATE DEFINITION
# ─────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    original_question: str  # For tracking rewrites
    context: str
    answer: str
    retrieval_time: float
    llm_time: float
    complexity: str  # SIMPLE, MEDIUM, COMPLEX
    relevance_score: float  # 0.0-1.0
    retry_count: int  # Track rewrite attempts

# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('rag_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# METRICS TRACKING
# ─────────────────────────────────────────────────────────────
class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retrieval_time = 0.0
        self.total_llm_time = 0.0
        self.total_response_time = 0.0
        self.start_time = datetime.now()
    
    def record_request(self, success: bool, retrieval_time: float, llm_time: float, total_time: float):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_retrieval_time += retrieval_time
        self.total_llm_time += llm_time
        self.total_response_time += total_time
    
    def get_stats(self):
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.successful_requests / self.total_requests * 100, 2) if self.total_requests > 0 else 0,
            "avg_retrieval_time": round(self.total_retrieval_time / self.total_requests, 3) if self.total_requests > 0 else 0,
            "avg_llm_time": round(self.total_llm_time / self.total_requests, 3) if self.total_requests > 0 else 0,
            "avg_response_time": round(self.total_response_time / self.total_requests, 3) if self.total_requests > 0 else 0,
            "requests_per_second": round(self.total_requests / uptime, 2) if uptime > 0 else 0,
            "uptime_seconds": round(uptime, 2)
        }

metrics = Metrics()

# ─────────────────────────────────────────────────────────────
# LIFESPAN (Startup/Shutdown)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API Server...")
    logger.info("Model: Qwen/Qwen2.5-3B-Instruct")
    logger.info("Vector DB: Qdrant")
    logger.info("vLLM Endpoint: http://localhost:8082/v1")
    
    # Ensure collection exists first
    ensure_qdrant_collection(reset=False)
    
    # Auto-index if Qdrant is empty and PDFs exist
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        doc_count = collection_info.points_count
        logger.info(f"Qdrant collection status: {doc_count} documents")
        
        if doc_count == 0:
            # Check if PDF files exist in data/ folder
            pdf_files = glob.glob(os.path.join("data", "*.pdf"))
            if pdf_files:
                logger.info(f"Qdrant is empty. Auto-indexing {len(pdf_files)} PDF file(s)...")
                req = IndexRequest(
                    pdf_folder="data",
                    glob_pattern="*.pdf",
                    chunk_size=800,
                    chunk_overlap=120,
                    reset_collection=False
                )
                result = index_pdfs_sync(req)
                if result.get("chunks_indexed", 0) > 0:
                    logger.info(f"Auto-indexing complete: {result['chunks_indexed']} chunks indexed from {result['files_indexed']} files")
                else:
                    logger.warning(f"Auto-indexing found no PDFs to index")
            else:
                logger.info("Qdrant is empty but no PDF files found in data/ folder. Use /index endpoint to add documents.")
        else:
            logger.info(f"Qdrant already contains {doc_count} documents. Skipping auto-indexing.")
    except Exception as e:
        logger.warning(f"Auto-indexing check failed: {e}. You may need to manually index PDFs via /index endpoint.")
    
    yield
    logger.info("Shutting down RAG API Server...")

app = FastAPI(
    title="RAG API with vLLM",
    description="Production-ready RAG API with detailed logging and metrics",
    version="2.1.0",
    lifespan=lifespan
)

executor = ThreadPoolExecutor(max_workers=50)

# ─────────────────────────────────────────────────────────────
# EMBEDDINGS CONFIGURATION
# ─────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# BAAI/bge-m3 produces 1024-dimensional vectors
EMBEDDING_DIM = 1024

# ─────────────────────────────────────────────────────────────
# QDRANT VECTORSTORE CONFIGURATION
# ─────────────────────────────────────────────────────────────
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "my_documents"

# Initialize Qdrant client (on-disk storage)
qdrant_client = QdrantClient(path=QDRANT_PATH)

# Create collection if it doesn't exist
def ensure_qdrant_collection(reset: bool = False) -> None:
    """Ensure Qdrant collection exists (optionally reset)."""
    if reset:
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
            logger.warning(f"Deleted Qdrant collection (reset=true): {COLLECTION_NAME}")
        except Exception:
            # collection may not exist
            pass

    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"Loaded existing Qdrant collection: {COLLECTION_NAME}")
        return
    except Exception:
        logger.info(f"Creating new Qdrant collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

ensure_qdrant_collection(reset=False)

# Initialize vectorstore
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# ─────────────────────────────────────────────────────────────
# LLM CONFIGURATION (vLLM + Qwen)
# ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-3B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8082/v1",
    temperature=0.3,
    max_tokens=512,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a RAG assistant. Answer ONLY using the given CONTEXT.

CRITICAL RULES:
1. Use ONLY information from CONTEXT
2. Do NOT add external knowledge
3. Do NOT guess or hallucinate
4. If CONTEXT is empty or IRRELEVANT to the question, respond exactly with:
   "Answer not found in context."
5. Write the answer in Turkish using ASCII characters

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)

def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

# ─────────────────────────────────────────────────────────────
# QUERY PROCESSING
# ─────────────────────────────────────────────────────────────
class Query(BaseModel):
    question: str

class IndexRequest(BaseModel):
    pdf_folder: str = "data"
    glob_pattern: str = "*.pdf"
    max_files: int | None = None
    chunk_size: int = 800
    chunk_overlap: int = 120
    reset_collection: bool = False

def index_pdfs_sync(req: IndexRequest) -> dict:
    """Load PDFs, split into chunks, and upsert into Qdrant."""
    start_total = time.time()

    ensure_qdrant_collection(reset=req.reset_collection)

    pdf_files = sorted(glob.glob(os.path.join(req.pdf_folder, req.glob_pattern)))
    if req.max_files is not None:
        pdf_files = pdf_files[: req.max_files]

    if not pdf_files:
        return {
            "status": "ok",
            "message": "No PDF files found to index.",
            "pdf_folder": req.pdf_folder,
            "glob_pattern": req.glob_pattern,
            "files_indexed": 0,
            "pages_loaded": 0,
            "chunks_indexed": 0,
            "total_seconds": round(time.time() - start_total, 3),
        }

    pages = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        pages.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    ids = [str(uuid4()) for _ in range(len(chunks))]
    vectorstore.add_documents(documents=chunks, ids=ids)

    total_time = time.time() - start_total
    logger.info(
        f"Index OK | Files: {len(pdf_files)} | Pages: {len(pages)} | Chunks: {len(chunks)} | Total: {total_time:.3f}s"
    )

    return {
        "status": "ok",
        "pdf_folder": req.pdf_folder,
        "glob_pattern": req.glob_pattern,
        "files_indexed": len(pdf_files),
        "pages_loaded": len(pages),
        "chunks_indexed": len(chunks),
        "total_seconds": round(total_time, 3),
    }

# ─────────────────────────────────────────────────────────────
# LANGGRAPH NODES
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# LANGGRAPH NODES - ADAPTIVE & SELF-CORRECTION
# ─────────────────────────────────────────────────────────────

def classify_question(state: GraphState):
    """
    Classify question complexity: SIMPLE, MEDIUM, COMPLEX
    """
    question = state["question"]
    
    classify_prompt = f"""Classify this question's complexity. Reply with ONLY one word: SIMPLE, MEDIUM, or COMPLEX.

SIMPLE: Greetings, basic questions that don't need document retrieval (e.g., "Hello", "Thanks")
MEDIUM: Straightforward factual questions (e.g., "What is vLLM?")
COMPLEX: Multi-part or comparative questions (e.g., "Compare vLLM and TensorRT performance")

Question: {question}

Classification:"""
    
    try:
        response = llm.invoke(classify_prompt)
        complexity = response.content.strip().upper()
        
        # Validate response
        if complexity not in ["SIMPLE", "MEDIUM", "COMPLEX"]:
            complexity = "MEDIUM"  # Default fallback
        
        logger.info(f"Question classified as: {complexity} | Q: {question[:40]}...")
        return {"complexity": complexity}
    except Exception as e:
        logger.warning(f"Classification failed: {e}, defaulting to MEDIUM")
        return {"complexity": "MEDIUM"}

def direct_answer(state: GraphState):
    """
    Answer simple questions directly without retrieval (e.g., greetings)
    """
    start_time = time.time()
    question = state["question"]
    
    simple_prompt = f"""You are a helpful assistant. Answer this simple question briefly and politely.

Question: {question}

Answer:"""
    
    response = llm.invoke(simple_prompt)
    llm_time = time.time() - start_time
    
    logger.info(f"Direct answer (no retrieval) | LLM: {llm_time:.3f}s | Q: {question[:40]}...")
    
    return {
        "answer": response.content,
        "llm_time": llm_time,
        "retrieval_time": 0.0,
        "context": "",
        "relevance_score": 1.0
    }

def retrieve(state: GraphState):
    """
    Retrieve documents with adaptive k based on complexity.
    """
    start_time = time.time()
    question = state["question"]
    complexity = state.get("complexity", "MEDIUM")
    
    # Adaptive retrieval: adjust k based on complexity
    k_map = {"SIMPLE": 3, "MEDIUM": 6, "COMPLEX": 10}
    k = k_map.get(complexity, 6)
    
    # Update retriever with adaptive k
    adaptive_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    retrieved_docs = adaptive_retriever.invoke(question)
    retrieval_time = time.time() - start_time
    
    if not retrieved_docs:
        logger.warning(f"No docs found | Q: {question[:50]}... | Retrieval: {retrieval_time:.3f}s")
        return {
            "context": "",
            "retrieval_time": retrieval_time,
            "relevance_score": 0.0
        }
    
    context = format_docs(retrieved_docs)
    logger.info(f"Retrieved {len(retrieved_docs)} docs (k={k}) | Time: {retrieval_time:.3f}s")
    
    return {
        "context": context,
        "retrieval_time": retrieval_time
    }

def grade_documents(state: GraphState):
    """
    Grade document relevance using LLM.
    Returns relevance score (0.0-1.0).
    """
    question = state["question"]
    context = state["context"]
    
    if not context:
        return {"relevance_score": 0.0}
    
    grade_prompt = f"""You are a grading assistant. Evaluate if the CONTEXT is relevant to answer the QUESTION.

Reply with ONLY one word: RELEVANT or IRRELEVANT

QUESTION: {question}

CONTEXT:
{context[:1000]}...

Evaluation:"""
    
    try:
        response = llm.invoke(grade_prompt)
        grade = response.content.strip().upper()
        
        relevance_score = 1.0 if "RELEVANT" in grade else 0.0
        
        logger.info(f"Document grading: {grade} (score={relevance_score}) | Q: {question[:40]}...")
        return {"relevance_score": relevance_score}
    except Exception as e:
        logger.warning(f"Grading failed: {e}, assuming relevant")
        return {"relevance_score": 0.5}

def rewrite_query(state: GraphState):
    """
    Rewrite the query to improve retrieval quality.
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    
    rewrite_prompt = f"""You are a query optimization expert. Rewrite this question to be more specific and retrieval-friendly.

Original question: {question}

Rewritten question (be concise, keep the same language):"""
    
    try:
        response = llm.invoke(rewrite_prompt)
        rewritten = response.content.strip()
        
        logger.info(f"Query rewritten (attempt {retry_count + 1}): '{question[:30]}...' → '{rewritten[:30]}...'")
        
        return {
            "question": rewritten,
            "retry_count": retry_count + 1
        }
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return {"retry_count": retry_count + 1}

def generate(state: GraphState):
    """
    Generate answer using the LLM.
    """
    start_time = time.time()
    question = state["original_question"]  # Use original question for answer
    context = state["context"]
    
    # Early exit if no context found
    if not context:
        return {
            "answer": "Answer not found in context.",
            "llm_time": 0.0
        }

    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    llm_time = time.time() - start_time
    
    return {
        "answer": response.content,
        "llm_time": llm_time
    }

# ─────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def route_by_complexity(state: GraphState):
    """Route based on question complexity."""
    complexity = state.get("complexity", "MEDIUM")
    
    if complexity == "SIMPLE":
        return "direct_answer"
    else:
        return "retrieve"

def should_rewrite(state: GraphState):
    """Decide if we should rewrite query based on relevance."""
    relevance = state.get("relevance_score", 1.0)
    retry_count = state.get("retry_count", 0)
    
    # If documents are irrelevant and we haven't retried too many times
    if relevance < 0.5 and retry_count < 2:
        logger.info(f"Low relevance ({relevance}), rewriting query...")
        return "rewrite"
    else:
        return "generate"

# ─────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION - ADVANCED RAG
# ─────────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("classify", classify_question)
workflow.add_node("direct_answer", direct_answer)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate)

# Set entry point
workflow.set_entry_point("classify")

# Add conditional routing from classify
workflow.add_conditional_edges(
    "classify",
    route_by_complexity,
    {
        "direct_answer": "direct_answer",
        "retrieve": "retrieve"
    }
)

# Direct answer goes straight to END
workflow.add_edge("direct_answer", END)

# After retrieval, grade the documents
workflow.add_edge("retrieve", "grade")

# After grading, decide: rewrite or generate
workflow.add_conditional_edges(
    "grade",
    should_rewrite,
    {
        "rewrite": "rewrite",
        "generate": "generate"
    }
)

# After rewrite, try retrieval again
workflow.add_edge("rewrite", "retrieve")

# After generation, we're done
workflow.add_edge("generate", END)

# Compile graph
rag_graph = workflow.compile()

def process_query_sync(question: str) -> tuple[str, float, float, float]:
    """
    Synchronous query processing using LangGraph with adaptive retrieval and self-correction.
    Returns: (answer, retrieval_time, llm_time, total_time)
    """
    start_total = time.time()
    
    # Initialize state with all required fields
    initial_state = {
        "question": question,
        "original_question": question,  # Keep original for final answer
        "context": "",
        "answer": "",
        "retrieval_time": 0.0,
        "llm_time": 0.0,
        "complexity": "MEDIUM",
        "relevance_score": 1.0,
        "retry_count": 0
    }
    
    # Invoke graph
    try:
        final_state = rag_graph.invoke(initial_state)
        
        answer = final_state.get("answer", "Error occurred")
        retrieval_time = final_state.get("retrieval_time", 0.0)
        llm_time = final_state.get("llm_time", 0.0)
        total_time = time.time() - start_total
        
        # Log with additional metadata
        complexity = final_state.get("complexity", "UNKNOWN")
        relevance = final_state.get("relevance_score", 0.0)
        retries = final_state.get("retry_count", 0)
        
        logger.info(
            f"Query OK (Advanced RAG) | "
            f"Complexity: {complexity} | "
            f"Relevance: {relevance:.2f} | "
            f"Retries: {retries} | "
            f"Retrieval: {retrieval_time:.3f}s | "
            f"LLM: {llm_time:.3f}s | "
            f"Total: {total_time:.3f}s | "
            f"Q: {question[:40]}..."
        )
        
        return answer, retrieval_time, llm_time, total_time
        
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        return "An error occurred during graph execution.", 0.0, 0.0, time.time() - start_total

# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.post("/query")
async def query(q: Query):
    loop = asyncio.get_event_loop()
    try:
        answer, retrieval_time, llm_time, total_time = await loop.run_in_executor(
            executor, process_query_sync, q.question
        )
        metrics.record_request(True, retrieval_time, llm_time, total_time)
        return {
            "answer": answer,
            "metrics": {
                "retrieval_time": round(retrieval_time, 3),
                "llm_time": round(llm_time, 3),
                "total_time": round(total_time, 3)
            }
        }
    except Exception as e:
        logger.error(f"Query failed | Error: {str(e)} | Q: {q.question[:50]}...")
        metrics.record_request(False, 0, 0, 0)
        return {"answer": "An error occurred while processing your query.", "error": str(e)}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0"
    }

@app.get("/metrics")
def get_metrics():
    """Get detailed performance metrics"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics.get_stats()
    }

@app.get("/stats")
def get_stats():
    """Get vectorstore statistics"""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "status": "ok",
            "vectorstore": {
                "type": "Qdrant",
                "total_documents": collection_info.points_count,
                "collection_name": COLLECTION_NAME,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": str(collection_info.config.params.vectors.distance)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/index")
async def index_documents(req: IndexRequest):
    """
    Index PDF files into Qdrant.

    NOTE: This is an explicit step. If Qdrant is empty, retrieval will return no docs.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, index_pdfs_sync, req)
        return result
    except Exception as e:
        logger.error(f"Index failed | Error: {str(e)}")
        return {"status": "error", "message": str(e)}

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
