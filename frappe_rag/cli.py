from __future__ import annotations

import argparse
import os
from typing import List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from .chain import create_qa_chain, ensure_ascii_answer
from .config import AppConfig, load_config_file, merge_config
from .constants import DEFAULT_COLLECTION, DEFAULT_DATA_DIR, DEFAULT_QDRANT_PATH
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
from .server import run_server

load_dotenv()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG with local Qdrant + HF embeddings")
    parser.add_argument("--config", help="Path to YAML config", default=None)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--qdrant-path", default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)

    parser.add_argument("--index", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)

    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)

    parser.add_argument("-k", "--top-k", type=int, default=6)
    parser.add_argument("--retrieval", choices=["similarity", "mmr"], default="similarity")

    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--llm-backend", choices=["ollama", "vllm"], default="vllm")
    parser.add_argument("--llm-model", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--vllm-base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8282/v1"))
    parser.add_argument("--temperature", type=float, default=0.3)

    parser.add_argument("--enable-hybrid", action="store_true", default=True, help="Use dense + sparse hybrid search")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")

    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)

    defaults = parser.parse_args([])
    args = parser.parse_args(argv)
    args._defaults = defaults
    return args


def run_index(args: argparse.Namespace) -> int:
    cfg = merge_config(args, load_config_file(args.config))

    pdfs = find_pdfs(cfg.data_dir)
    if args.max_files and args.max_files > 0:
        pdfs = pdfs[: args.max_files]
    if not pdfs:
        print(f"No PDFs found in: {cfg.data_dir}")
        return 2

    device = pick_device(cfg.device)
    embeddings = create_embeddings(cfg.embedding_model, device)
    dim = embedding_dim(embeddings)

    docs = load_pdfs(pdfs)
    chunks = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)

    client = QdrantClient(path=cfg.qdrant_path)
    preexisted = ensure_collection(client, cfg.collection, dim, args.force, cfg.enable_hybrid)

    added, skipped = index_chunks(
        client,
        cfg.collection,
        chunks,
        args.force,
        preexisted,
        embeddings,
        cfg.enable_hybrid,
        compute_sparse_vector if cfg.enable_hybrid else None,
    )

    print(f"Indexed chunks: total={len(chunks)} added={added} skipped={skipped} device={device}")
    return 0


def auto_index_if_needed(cfg: AppConfig, embeddings, client: QdrantClient) -> bool:
    if collection_exists(cfg.qdrant_path, cfg.collection):
        print(f"Index found: {cfg.collection}")
        return True

    pdfs = find_pdfs(cfg.data_dir)
    if not pdfs:
        print(f"No PDFs in {cfg.data_dir}")
        return False

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
    return True


def run_chat(args: argparse.Namespace) -> int:
    cfg = merge_config(args, load_config_file(args.config))

    device = pick_device(cfg.device)
    embeddings = create_embeddings(cfg.embedding_model, device)
    client = QdrantClient(path=cfg.qdrant_path)

    if not auto_index_if_needed(cfg, embeddings, client):
        print("No data to index. Add PDFs to data/ folder.")
        return 2

    rerank_fn = CrossEncoderReranker(cfg.reranker_model, device) if cfg.enable_hybrid else None
    if cfg.enable_hybrid:
        retriever = HybridRetriever(client, embeddings, cfg.collection, cfg.top_k, rerank_fn)
    else:
        vectorstore = get_vectorstore(client, cfg.collection, embeddings)
        retriever = vectorstore.as_retriever(search_type=cfg.retrieval, search_kwargs={"k": cfg.top_k})

    llm = create_llm(cfg.llm_backend, cfg.llm_model, cfg.temperature, cfg.ollama_url, cfg.vllm_base_url)
    qa_chain = create_qa_chain(retriever, llm, cfg.top_k, cfg.retrieval)

    print("RAG ready. Type 'exit' to quit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            return 0

        try:
            raw = qa_chain.invoke(question)
            answer = ensure_ascii_answer(str(raw))
            print(f"Answer: {answer}\n")
        except Exception as e:
            print(f"Error: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.index:
        return run_index(args)
    if args.serve:
        cfg = merge_config(args, load_config_file(args.config))
        cfg.serve_host = args.host
        cfg.serve_port = args.port
        return run_server(cfg)
    return run_chat(args)
