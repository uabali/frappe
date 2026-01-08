from __future__ import annotations

import torch
from langchain_huggingface import HuggingFaceEmbeddings


def pick_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_embeddings(model_name: str, device: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dim(embeddings: HuggingFaceEmbeddings) -> int:
    return len(embeddings.embed_query("test"))
