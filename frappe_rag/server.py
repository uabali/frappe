from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from .chain import create_qa_chain, ensure_ascii_answer
from .config import AppConfig
from .embeddings import create_embeddings, embedding_dim, pick_device
from .llm import create_llm
from .pdf_ingest import find_pdfs, load_pdfs, split_documents
from .qdrant_store import (
    HybridRetriever,
    collection_exists,
    compute_sparse_vector,
    ensure_collection,
    get_vectorstore,
    index_chunks,
)
from .reranker import CrossEncoderReranker


@dataclass
class ServerMetrics:
    total_requests: int = 0
    total_successful: int = 0
    total_failed: int = 0
    last_latency_ms: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_llm_time: float = 0.0

    def record(self, latency_ms: float, retrieval_time: float, llm_time: float, success: bool) -> None:
        self.total_requests += 1
        if success:
            self.total_successful += 1
        else:
            self.total_failed += 1
        self.last_latency_ms = latency_ms
        # Running average
        n = self.total_successful
        if n > 0 and success:
            self.avg_retrieval_time = ((self.avg_retrieval_time * (n - 1)) + retrieval_time) / n
            self.avg_llm_time = ((self.avg_llm_time * (n - 1)) + llm_time) / n


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryMetricsResponse(BaseModel):
    retrieval_time: float
    llm_time: float
    total_time: float


class QueryResponse(BaseModel):
    answer: str
    metrics: QueryMetricsResponse


def create_app(cfg: AppConfig, qa_chain, retriever) -> FastAPI:
    app = FastAPI(title="Frappe RAG", version="0.1.0")
    server_metrics = ServerMetrics()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics_endpoint():
        return asdict(server_metrics)

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        if req.top_k and hasattr(retriever, "top_k"):
            retriever.top_k = req.top_k

        started = time.perf_counter()
        retrieval_start = time.perf_counter()
        try:
            # Retrieval step
            docs = retriever.invoke(req.question) if hasattr(retriever, "invoke") else retriever._get_relevant_documents(req.question)
            retrieval_time = time.perf_counter() - retrieval_start

            # LLM step
            llm_start = time.perf_counter()
            result = await qa_chain.ainvoke(req.question)
            llm_time = time.perf_counter() - llm_start

            total_time = time.perf_counter() - started
            server_metrics.record(total_time * 1000, retrieval_time, llm_time, success=True)

            return QueryResponse(
                answer=ensure_ascii_answer(str(result)),
                metrics=QueryMetricsResponse(
                    retrieval_time=round(retrieval_time, 4),
                    llm_time=round(llm_time, 4),
                    total_time=round(total_time, 4),
                ),
            )
        except Exception as e:
            total_time = time.perf_counter() - started
            server_metrics.record(total_time * 1000, 0, 0, success=False)
            return QueryResponse(
                answer=f"Error: {str(e)}",
                metrics=QueryMetricsResponse(retrieval_time=0, llm_time=0, total_time=round(total_time, 4)),
            )

    return app


def auto_index_if_needed(cfg: AppConfig, embeddings, client: QdrantClient) -> None:
    if collection_exists(cfg.qdrant_path, cfg.collection):
        print(f"Index found: {cfg.collection}")
        return

    pdfs = find_pdfs(cfg.data_dir)
    if not pdfs:
        print(f"No PDFs in {cfg.data_dir}, skipping indexing")
        return

    print(f"Auto-indexing {len(pdfs)} PDFs...")
    dim = embedding_dim(embeddings)
    docs = load_pdfs(pdfs)
    chunks = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)

    ensure_collection(client, cfg.collection, dim, False, cfg.enable_hybrid)
    added, skipped = index_chunks(
        client,
        cfg.collection,
        chunks,
        False,
        False,
        embeddings,
        cfg.enable_hybrid,
        compute_sparse_vector if cfg.enable_hybrid else None,
    )
    print(f"Indexed: added={added} skipped={skipped}")


def run_server(cfg: AppConfig) -> int:
    device = pick_device(cfg.device)
    embeddings = create_embeddings(cfg.embedding_model, device)
    client = QdrantClient(path=cfg.qdrant_path)

    auto_index_if_needed(cfg, embeddings, client)

    rerank_fn = CrossEncoderReranker(cfg.reranker_model, device) if cfg.enable_hybrid else None
    if cfg.enable_hybrid:
        retriever = HybridRetriever(client, embeddings, cfg.collection, cfg.top_k, rerank_fn)
    else:
        vectorstore = get_vectorstore(client, cfg.collection, embeddings)
        retriever = vectorstore.as_retriever(search_type=cfg.retrieval, search_kwargs={"k": cfg.top_k})

    llm = create_llm(cfg.llm_backend, cfg.llm_model, cfg.temperature, cfg.ollama_url, cfg.vllm_base_url)
    qa_chain = create_qa_chain(retriever, llm, cfg.top_k, cfg.retrieval)

    app = create_app(cfg, qa_chain, retriever)
    uvicorn.run(app, host=cfg.serve_host, port=cfg.serve_port)
    return 0
