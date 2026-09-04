"""Configuration loading for the isolated LangChain baseline."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


class EmbeddingConfig(BaseModel):
    model: str
    base_url: str
    api_key: str
    dimensions: int = 1024
    timeout_seconds: float = 60
    max_retries: int = 2
    batch_size: int = 10


class RerankConfig(BaseModel):
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float = 60
    max_retries: int = 2


class MilvusConfig(BaseModel):
    uri: str = "http://localhost:19530"
    db: str = "oncall"
    collection: str = "langchain_chunks"


class ChunkingConfig(BaseModel):
    parent_tokens: int = 900
    parent_overlap: int = 100
    child_tokens: int = 250
    child_overlap: int = 40


class RetrievalConfig(BaseModel):
    candidate_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 8


class Settings(BaseModel):
    embedding: EmbeddingConfig
    rerank: RerankConfig
    milvus: MilvusConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def _expand_env(value: Any) -> Any:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        return default if default is not None else match.group(0)

    if isinstance(value, str):
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_settings(path: str | Path | None = None) -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(
        path or os.environ.get("MINIRAG_LANGCHAIN_CONFIG") or project_root / "config.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"configuration not found: {config_path}; copy config.example.yaml "
            "to config.yaml or set MINIRAG_LANGCHAIN_CONFIG"
        )
    load_dotenv(config_path.parent / ".env", override=False)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    # The baseline can reuse MiniRAG's private config, which has compatible
    # embedding/rerank/milvus sections but no collection name.
    milvus = raw.setdefault("milvus", {})
    milvus.setdefault(
        "collection",
        os.environ.get("MINIRAG_LANGCHAIN_COLLECTION", "langchain_chunks"),
    )
    return Settings.model_validate(_expand_env(raw))
