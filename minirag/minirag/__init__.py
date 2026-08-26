"""minirag：精简轻量版 LightRAG 检索内核（Milvus + PostgreSQL，检索即服务）。"""
from minirag.rag import MiniRAG
from minirag.schemas import (
    DocumentInput,
    Evidence,
    IndexReport,
    Keywords,
    QueryParam,
    RetrievalResult,
)

__all__ = [
    "MiniRAG",
    "QueryParam",
    "DocumentInput",
    "RetrievalResult",
    "IndexReport",
    "Evidence",
    "Keywords",
]

__version__ = "0.1.0"
