import os
import re
import glob
import time
import json
import hashlib
import torch
import logging
import numpy as np
from uuid import uuid4
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    SparseVector, Filter, FieldCondition, MatchValue,
    PointStruct, Prefetch, FusionQuery
)
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"

load_dotenv()

VLLM_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8282/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "casperhansen/llama-3-8b-instruct-awq")
COLLECTION = "documents"
TOTAL_MAX_TOKENS = 3500
PROMPT_OVERHEAD = 300
HYDE_WEIGHT = float(os.getenv("HYDE_WEIGHT", "0.4"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "20"))
RERANK_THRESHOLD_BASE = float(os.getenv("RERANK_THRESHOLD_BASE", "-6.0"))
MAX_ANSWER_HISTORY_TOKENS = int(os.getenv("MAX_ANSWER_HISTORY_TOKENS", "200"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

ENABLE_GENERAL_FALLBACK = os.getenv("ENABLE_GENERAL_FALLBACK", "true").lower() == "true"
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_PARENT_RETRIEVER = os.getenv("ENABLE_PARENT_RETRIEVER", "true").lower() == "true"
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "200"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1200"))

try:
    tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL)
    logger.info(f"Loaded tokenizer: {VLLM_MODEL}")
except Exception as e:
    logger.warning(f"Failed to load tokenizer: {e}")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_text(text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True) + "..."

def normalize_turkish(text: str) -> str:
    text = text.lower()
    replacements = {'ı': 'i', 'İ': 'i', 'ğ': 'g', 'Ğ': 'g', 'ü': 'u', 'Ü': 'u',
                   'ş': 's', 'Ş': 's', 'ö': 'o', 'Ö': 'o', 'ç': 'c', 'Ç': 'c'}
    for tr, en in replacements.items():
        text = text.replace(tr, en)
    return text

def get_doc_weight(metadata: Dict) -> float:
    if 'doc_weight' in metadata:
        return metadata['doc_weight']
    return 1.0

def compute_dynamic_threshold(query: str, docs: List) -> float:
    base = RERANK_THRESHOLD_BASE
    query_tokens = count_tokens(query)
    if query_tokens > 20:
        base += 0.5
    elif query_tokens < 5:
        base -= 1.0
    if len(docs) < 5:
        base -= 1.5
    elif len(docs) > 15:
        base += 0.5
    return base

def compute_sparse_vector(text: str) -> Dict[str, Any]:
    words = re.findall(r'\b\w+\b', text.lower())
    tf = {}
    for word in words:
        if len(word) > 2:
            tf[word] = tf.get(word, 0) + 1
    if not tf:
        return {"indices": [], "values": []}
    
    # Use a dictionary to aggregate values for colliding indices
    vector_map = {}
    max_tf = max(tf.values())
    for word, count in tf.items():
        idx = abs(hash(word) % 100000)
        val = np.log(1 + count / max_tf)
        vector_map[idx] = vector_map.get(idx, 0.0) + val
        
    return {"indices": list(vector_map.keys()), "values": list(vector_map.values())}

def clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text

def get_files_hash(file_list: List[str]) -> str:
    hasher = hashlib.md5()
    for file_path in sorted(file_list):
        stats = os.stat(file_path)
        hasher.update(f"{file_path}{stats.st_mtime}".encode("utf-8"))
    return hasher.hexdigest()

embed_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embeddings device: {embed_device}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": embed_device},
    encode_kwargs={"normalize_embeddings": True}
)

client = QdrantClient(path="./qdrant_db")
parent_store: Dict[str, str] = {}

def index_documents():
    global parent_store
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        print("No PDF files found in data/")
        return

    cache_file = "data/.index_cache"
    settings_hash = f"_{CHUNK_SIZE}_{CHUNK_OVERLAP}_{ENABLE_HYBRID_SEARCH}_{ENABLE_PARENT_RETRIEVER}"
    current_hash = get_files_hash(pdf_files) + settings_hash

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = f.read().strip()
        if cached == current_hash:
            try:
                info = client.get_collection(COLLECTION)
                if info.points_count > 0:
                    print(f"Index up-to-date ({info.points_count} chunks). Skipping.")
                    if os.path.exists("data/.parent_store.json"):
                        with open("data/.parent_store.json", "r") as f:
                            parent_store = json.load(f)
                    return
            except:
                pass

    print(f"Indexing {len(pdf_files)} PDF(s)...")
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        filename = os.path.basename(pdf)
        
        for d in docs:
            d.page_content = clean_text(d.page_content)
            d.page_content = f"[Source: {filename}]\n{d.page_content}"
        documents.extend(docs)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    vectors_config = {"dense": VectorParams(size=1024, distance=Distance.COSINE)}
    sparse_config = None
    if ENABLE_HYBRID_SEARCH:
        sparse_config = {"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_config
    )

    if ENABLE_PARENT_RETRIEVER:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        parent_chunks = parent_splitter.split_documents(documents)
        all_child_chunks = []
        parent_store = {}
        
        for parent in parent_chunks:
            parent_id = str(uuid4())
            children = child_splitter.split_documents([parent])
            for child in children:
                child_id = str(uuid4())
                child.metadata['child_id'] = child_id
                child.metadata['parent_id'] = parent_id
                parent_store[child_id] = parent.page_content
                all_child_chunks.append((child_id, child))
        
        chunks_to_index = all_child_chunks
        with open("data/.parent_store.json", "w") as f:
            json.dump(parent_store, f)
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        chunks_to_index = [(str(uuid4()), c) for c in chunks]

    print(f"Embedding {len(chunks_to_index)} chunks...")
    points = []
    batch_size = 32
    
    for i in range(0, len(chunks_to_index), batch_size):
        batch = chunks_to_index[i:i+batch_size]
        texts = [c.page_content for _, c in batch]
        dense_vectors = embeddings.embed_documents(texts)
        
        for j, (chunk_id, chunk) in enumerate(batch):
            point_vectors = {"dense": dense_vectors[j]}
            if ENABLE_HYBRID_SEARCH:
                sparse = compute_sparse_vector(chunk.page_content)
                point_vectors["sparse"] = SparseVector(indices=sparse["indices"], values=sparse["values"])
            
            points.append(PointStruct(
                id=chunk_id,
                vector=point_vectors,
                payload={"page_content": chunk.page_content, "metadata": chunk.metadata}
            ))
    
    for i in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION, points=points[i:i+100])

    with open(cache_file, "w") as f:
        f.write(current_hash)
    
    features = []
    if ENABLE_HYBRID_SEARCH: features.append("Hybrid")
    if ENABLE_PARENT_RETRIEVER: features.append("Parent-Doc")
    print(f"Indexing Done! Features: {', '.join(features) if features else 'Standard'}")

def hybrid_search(query: str, k: int = RETRIEVER_K) -> List[Document]:
    dense_query = embeddings.embed_query(query)
    
    if ENABLE_HYBRID_SEARCH:
        sparse_query = compute_sparse_vector(query)
        try:
            results = client.query_points(
                collection_name=COLLECTION,
                prefetch=[
                    Prefetch(query=dense_query, using="dense", limit=k * 2),
                    Prefetch(
                        query=SparseVector(indices=sparse_query["indices"], values=sparse_query["values"]),
                        using="sparse", limit=k * 2
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=k,
                with_payload=True
            )
            
            docs = []
            for point in results.points:
                content = point.payload.get("page_content", "")
                metadata = point.payload.get("metadata", {})
                if ENABLE_PARENT_RETRIEVER and "child_id" in metadata:
                    child_id = metadata["child_id"]
                    if child_id in parent_store:
                        content = parent_store[child_id]
                        metadata["retrieved_as"] = "parent"
                docs.append(Document(page_content=content, metadata=metadata))
            return docs
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
    
    # Fallback to dense only
    results = client.search(
        collection_name=COLLECTION,
        query_vector=("dense", dense_query),
        limit=k,
        with_payload=True
    )
    
    docs = []
    for point in results:
        content = point.payload.get("page_content", "")
        metadata = point.payload.get("metadata", {})
        if ENABLE_PARENT_RETRIEVER and "child_id" in metadata:
            if metadata["child_id"] in parent_store:
                content = parent_store[metadata["child_id"]]
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

index_documents()

device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=device)

llm = ChatOpenAI(
    base_url=VLLM_URL, api_key="EMPTY", model=VLLM_MODEL,
    temperature=0.3, max_tokens=512
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant answering questions based on the provided context.

Rules:
- Quote directly when possible
- Synthesize from multiple sources
- If info is missing, say so

Always cite sources like [filename.pdf - p1]. Write in Turkish.

Context:
{context}"""),
    ("human", "History:\n{history}\n\nQuestion: {question}")
])

PROMPT_GENERAL_FALLBACK = ChatPromptTemplate.from_messages([
    ("system", """Answer based on general knowledge, not documents.
Start with: "📚 Bu bilgi dokümanlarınızda bulunamadı. Genel bilgi olarak:" """),
    ("human", "{question}")
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Write a short ideal answer to this question in Turkish:\nQuestion: {question}\nIdeal Answer:")
])

def rerank_with_source(query: str, docs: List, hyde_answer: Optional[str] = None, max_tokens: int = 2500):
    if not docs:
        return None, [], []

    threshold = compute_dynamic_threshold(query, docs)
    scores_q = reranker.predict([[query, d.page_content] for d in docs])

    if hyde_answer:
        scores_h = reranker.predict([[hyde_answer, d.page_content] for d in docs])
        scores = (1 - HYDE_WEIGHT) * scores_q + HYDE_WEIGHT * scores_h
    else:
        scores = scores_q.tolist() if hasattr(scores_q, "tolist") else list(scores_q)

    weighted_scores = [s * get_doc_weight(d.metadata) for d, s in zip(docs, scores)]
    ranked = sorted(zip(docs, weighted_scores), key=lambda x: x[1], reverse=True)
    ranked = [(d, s) for d, s in ranked if s >= threshold]

    if not ranked:
        return None, [], []

    result_parts, sources_used, chunks_used = [], [], []
    total_tokens = 0

    for d, score in ranked:
        source = os.path.basename(d.metadata.get("source", "?"))
        page = d.metadata.get("page", 0) + 1
        part = f"[{source} - p{page}] (score: {score:.2f})\n{d.page_content}"
        part_tokens = count_tokens(part)
        if total_tokens + part_tokens > max_tokens:
            break
        result_parts.append(part)
        sources_used.append(f"{source} (p{page})")
        chunks_used.append((d, score))
        total_tokens += part_tokens

    return ("\n\n---\n\n".join(result_parts), sources_used, chunks_used) if result_parts else (None, [], [])

def format_history(history: List, max_tokens: int = 800) -> tuple:
    if not history:
        return "None", 0
    result, total = [], 0
    for h in reversed(history):
        answer = truncate_text(h['a'], MAX_ANSWER_HISTORY_TOKENS)
        entry = f"Q: {h['q']}\nA: {answer}"
        tokens = count_tokens(entry)
        if total + tokens > max_tokens:
            break
        result.insert(0, entry)
        total += tokens
    return ("\n".join(result), total) if result else ("None", 0)

def hyde_search(question: str):
    hyde_answer = llm.invoke(HYDE_PROMPT.invoke({"question": question}), max_tokens=128).content.strip()
    norm_question = normalize_turkish(question)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(hybrid_search, hyde_answer, RETRIEVER_K)
        f2 = executor.submit(hybrid_search, question, RETRIEVER_K)
        f3 = executor.submit(hybrid_search, norm_question, RETRIEVER_K)
        docs_hyde, docs_orig, docs_norm = f1.result(), f2.result(), f3.result()

    seen, combined = set(), []
    for d in docs_hyde + docs_orig + docs_norm:
        key = f"{d.metadata.get('source')}_{d.metadata.get('page')}_{hash(d.page_content[:100])}"
        if key not in seen:
            seen.add(key)
            combined.append(d)
    return combined, hyde_answer

print("\n=== FRAPPE RAG (Advanced) ===")
features = []
if ENABLE_HYBRID_SEARCH: features.append("Hybrid Search")
if ENABLE_PARENT_RETRIEVER: features.append("Parent-Doc Retriever")
print(f"Features: {', '.join(features) if features else 'Standard'}")
print("Commands: exit\n")

history = []
debug = False
use_hyde = True
allow_fallback = ENABLE_GENERAL_FALLBACK

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not q:
        continue
    cmd = q.lower()
    
    if cmd == "exit":
        break

    t0 = time.time()
    try:
        hyde_answer = None
        if use_hyde:
            docs, hyde_answer = hyde_search(q)
        else:
            docs = hybrid_search(q, RETRIEVER_K)

        t1 = time.time()

        if not docs:
            if allow_fallback:
                print("\n⚠️ Dokümanlarda bilgi bulunamadı.")
                if input("Genel bilgi? (e/h): ").strip().lower() == 'e':
                    resp = llm.invoke(PROMPT_GENERAL_FALLBACK.invoke({"question": q})).content.strip()
                    print(f"\nA: {resp}\n")
                    continue
            print("\nA: Bu bilgi dokümanlarda yok.\n")
            continue

        history_str, history_tokens = format_history(history)
        available_ctx = TOTAL_MAX_TOKENS - history_tokens - count_tokens(q) - PROMPT_OVERHEAD - 512
        available_ctx = max(500, available_ctx)

        context, sources, chunks = rerank_with_source(q, docs, hyde_answer, available_ctx)
        t2 = time.time()

        if context is None:
            if allow_fallback:
                print("\n⚠️ Yeterli bilgi bulunamadı.")
                if input("Genel bilgi? (e/h): ").strip().lower() == 'e':
                    resp = llm.invoke(PROMPT_GENERAL_FALLBACK.invoke({"question": q})).content.strip()
                    print(f"\nA: {resp}\n")
                    history.append({"q": q, "a": resp})
                    continue
            print("\nA: Bu bilgi dokümanlarda yok.\n")
            continue

        if debug:
            print(f"[DEBUG] Retrieval: {t1-t0:.2f}s | Rerank: {t2-t1:.2f}s | Docs: {len(docs)}")

        prompt = PROMPT.invoke({
            "context": context, "question": q, "history": history_str
        })

        gen_tokens = min(512, max(256, TOTAL_MAX_TOKENS - count_tokens(str(prompt))))
        response_text = llm.invoke(prompt, max_tokens=gen_tokens).content.strip()
        t3 = time.time()

        print(f"\nA: {response_text}")
        
        # Simple citation printing
        unique_sources = list(dict.fromkeys(sources))[:5]
        if unique_sources:
            print(f"📄 Kaynaklar: {', '.join(unique_sources)}")
            
        if debug:
            print(f"[DEBUG] Gen: {t3-t2:.2f}s | Total: {t3-t0:.2f}s")
        print()

        history.append({"q": q, "a": response_text})

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\nError: {e}\n")
