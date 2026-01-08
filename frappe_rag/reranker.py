from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str) -> None:
        self.encoder = CrossEncoder(model_name, max_length=512, device=device)

    def __call__(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return docs
        pairs = [[query, d.page_content] for d in docs]
        scores = self.encoder.predict(pairs)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in scored[:top_k]]
