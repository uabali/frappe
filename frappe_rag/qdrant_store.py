from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FusionQuery,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


def ensure_collection(
    client: QdrantClient,
    collection: str,
    dim: int,
    force: bool,
    enable_sparse: bool,
) -> bool:
    existed = False
    try:
        client.get_collection(collection)
        existed = True
    except Exception:
        existed = False

    if force and existed:
        client.delete_collection(collection)
        existed = False

    if not existed:
        vectors_config = {"dense": VectorParams(size=dim, distance=Distance.COSINE)}
        sparse_config = {"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))} if enable_sparse else None
        client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

    return existed


def get_vectorstore(client: QdrantClient, collection: str, embeddings) -> QdrantVectorStore:
    return QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings, vector_name="dense")


def index_chunks(
    client: QdrantClient,
    collection: str,
    chunks: List[Document],
    force: bool,
    collection_preexisted: bool,
    embeddings,
    enable_sparse: bool,
    sparse_fn: Optional[Callable[[str], Dict[str, List[int]]]] = None,
) -> Tuple[int, int]:
    ids = [c.metadata.get("content_hash") for c in chunks if c.metadata.get("content_hash")]

    if force or not collection_preexisted:
        dense_vectors = embeddings.embed_documents([c.page_content for c in chunks])
        payloads = []
        for chunk, dense in zip(chunks, dense_vectors):
            c_id = chunk.metadata.get("content_hash")
            if not c_id:
                continue
            payloads.append(_to_point(c_id, chunk, dense, enable_sparse, sparse_fn))
        client.upsert(collection_name=collection, points=payloads)
        return len(chunks), 0

    existing_ids: set[str] = set()
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        try:
            res = client.retrieve(
                collection_name=collection,
                ids=batch,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids.update(str(p.id) for p in res)
        except Exception:
            continue

    new_chunks: List[Document] = []
    for chunk in chunks:
        c_id = chunk.metadata.get("content_hash")
        if not c_id or c_id in existing_ids:
            continue
        existing_ids.add(c_id)
        new_chunks.append(chunk)

    if not new_chunks:
        return 0, len(chunks) - len(new_chunks)

    dense_vectors = embeddings.embed_documents([c.page_content for c in new_chunks])
    payloads = []
    for chunk, dense in zip(new_chunks, dense_vectors):
        c_id = chunk.metadata.get("content_hash")
        if not c_id:
            continue
        payloads.append(_to_point(c_id, chunk, dense, enable_sparse, sparse_fn))

    client.upsert(collection_name=collection, points=payloads)
    skipped = len(chunks) - len(new_chunks)
    return len(new_chunks), skipped


def collection_exists(qdrant_path: str, collection: str) -> bool:
    client = QdrantClient(path=qdrant_path)
    try:
        client.get_collection(collection)
        return True
    except Exception:
        return False


def compute_sparse_vector(text: str) -> Dict[str, List[float]]:
    words = re.findall(r"\b\w+\b", text.lower())
    tf: Dict[str, int] = {}
    for word in words:
        if len(word) <= 2:
            continue
        tf[word] = tf.get(word, 0) + 1

    if not tf:
        return {"indices": [], "values": []}

    vector_map: Dict[int, float] = {}
    max_tf = max(tf.values())
    for word, count in tf.items():
        idx = abs(hash(word) % 100000)
        val = math.log1p(count / max_tf)
        vector_map[idx] = vector_map.get(idx, 0.0) + val

    return {"indices": list(vector_map.keys()), "values": list(vector_map.values())}


def _to_point(point_id: str, chunk: Document, dense_vector, enable_sparse: bool, sparse_fn):
    vectors = {"dense": dense_vector}
    if enable_sparse and sparse_fn:
        sparse = sparse_fn(chunk.page_content)
        vectors["sparse"] = SparseVector(indices=sparse["indices"], values=sparse["values"])
    return {
        "id": point_id,
        "vector": vectors,
        "payload": {"page_content": chunk.page_content, "metadata": chunk.metadata},
    }


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        client: QdrantClient,
        embeddings,
        collection: str,
        top_k: int,
        rerank_fn: Optional[Callable[[str, List[Document], int], List[Document]]] = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.embeddings = embeddings
        self.collection = collection
        self.top_k = top_k
        self.rerank_fn = rerank_fn

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_query = self.embeddings.embed_query(query)
        sparse_query = compute_sparse_vector(query)

        try:
            res = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    Prefetch(query=dense_query, using="dense", limit=self.top_k * 2),
                    Prefetch(
                        query=SparseVector(indices=sparse_query["indices"], values=sparse_query["values"]),
                        using="sparse",
                        limit=self.top_k * 2,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=self.top_k,
                with_payload=True,
            )
        except Exception:
            return []

        docs = [
            Document(page_content=p.payload.get("page_content", ""), metadata=p.payload.get("metadata", {}))
            for p in res.points
        ]

        if self.rerank_fn:
            docs = self.rerank_fn(query, docs, self.top_k)
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
