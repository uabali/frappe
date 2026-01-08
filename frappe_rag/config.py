from __future__ import annotations

import os
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional

import yaml

from .constants import DEFAULT_COLLECTION, DEFAULT_DATA_DIR, DEFAULT_QDRANT_PATH


@dataclass
class AppConfig:
    data_dir: str = DEFAULT_DATA_DIR
    qdrant_path: str = DEFAULT_QDRANT_PATH
    collection: str = DEFAULT_COLLECTION
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 6
    retrieval: str = "similarity"
    embedding_model: str = "BAAI/bge-m3"
    device: str = "auto"
    llm_backend: str = "vllm"
    llm_model: str = "Qwen2.5-3B-Instruct"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8282/v1")
    temperature: float = 0.3
    enable_hybrid: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    serve_host: str = "0.0.0.0"
    serve_port: int = 8088


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if v is not None}


def merge_config(args, config_data: Dict[str, Any]) -> AppConfig:
    base = asdict(AppConfig())
    base.update(config_data)

    defaults = getattr(args, "_defaults", None)
    default_map = asdict(AppConfig()) if defaults is None else vars(defaults)

    for field in fields(AppConfig):
        name = field.name
        arg_val = getattr(args, name, None)
        default_val = default_map.get(name)
        if arg_val is None:
            continue
        if arg_val != default_val:
            base[name] = arg_val
        elif name in config_data:
            base[name] = config_data[name]

    return AppConfig(**base)
