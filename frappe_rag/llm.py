from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


LlmBackend = Literal["ollama", "vllm"]


def create_llm(backend: LlmBackend, model: str, temperature: float, ollama_url: str, vllm_base_url: str):
    if backend == "vllm":
        return ChatOpenAI(base_url=vllm_base_url, api_key="EMPTY", model=model, temperature=temperature)
    return ChatOllama(model=model, base_url=ollama_url, temperature=temperature)
