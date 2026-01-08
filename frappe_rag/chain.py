from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .constants import FALLBACK_ANSWER
from .text_utils import to_ascii_turkish


def build_prompt() -> ChatPromptTemplate:
    template = """You are a retrieval-based assistant.

Answer the user's question ONLY using the given CONTEXT.
Do not use external knowledge.
Do not guess.

Rules:
- Write the answer in Turkish using ONLY ASCII characters.
- Do NOT use Turkish characters like: ç, ğ, ş, ı, İ, ö, ü.
- Keep the answer short, clear, and direct.
- If the answer is not found in the context, respond exactly with: \"Baglamda cevap bulunamadi.\"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    return ChatPromptTemplate.from_template(template)


def format_docs(docs: Iterable[Document]) -> str:
    parts: list[str] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        page_str = str(page + 1) if isinstance(page, int) else "?"
        parts.append(f"[source={src} page={page_str}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def create_qa_chain(retriever, llm, top_k: int, retrieval: str):
    prompt = build_prompt()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def ensure_ascii_answer(text: str) -> str:
    if not text:
        return FALLBACK_ANSWER
    if FALLBACK_ANSWER in text:
        return FALLBACK_ANSWER
    normalized = to_ascii_turkish(text)
    if not normalized:
        return FALLBACK_ANSWER
    if FALLBACK_ANSWER in normalized:
        return FALLBACK_ANSWER
    if not normalized.isascii():
        return FALLBACK_ANSWER
    return normalized
