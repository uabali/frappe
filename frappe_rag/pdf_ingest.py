from __future__ import annotations

import os
from glob import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .hashing import compute_content_hash
from .text_utils import clean_text


def find_pdfs(data_dir: str) -> List[str]:
    return sorted(glob(os.path.join(data_dir, "*.pdf")))


def load_pdfs(pdf_files: List[str]) -> List[Document]:
    docs: List[Document] = []
    for path in pdf_files:
        loader = PyPDFLoader(path)
        loaded = loader.load()
        filename = os.path.basename(path)
        for d in loaded:
            d.page_content = clean_text(d.page_content)
            d.metadata["source"] = filename
        docs.extend(loaded)
    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        chunk.metadata["content_hash"] = compute_content_hash(chunk.page_content, f"{source}:{page}")
    return chunks
