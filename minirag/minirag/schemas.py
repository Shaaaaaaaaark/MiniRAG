"""MiniRAG 的文档、分块和检索契约。"""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

BlockType = Literal["heading", "paragraph", "table"]


class Block(BaseModel):
    type: BlockType
    text: str
    level: int | None = None
    block_id: str | None = None


class DocumentInput(BaseModel):
    source: str
    source_id: str | None = None
    revision: str | None = None
    title: str | None = None
    text: str | None = None
    blocks: list[Block] | None = None


class ParsedDocument(BaseModel):
    document_id: str
    source: str
    source_id: str | None = None
    title: str | None = None
    revision: str | None = None
    blocks: list[Block] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    document_id: str
    ord: int
    heading_path: str
    content: str
    token_count: int
    parent_id: str | None = None
    parent_content: str | None = None
    block_id: str | None = None


class IndexReport(BaseModel):
    document_id: str
    chunks: int = 0


def make_evidence_id(ref_id: str) -> str:
    """由父块 ID 生成稳定证据 ID。"""
    return "e_" + hashlib.sha1(f"chunk:{ref_id}".encode()).hexdigest()[:16]


class Evidence(BaseModel):
    evidence_id: str
    kind: Literal["chunk"] = "chunk"
    ref_id: str
    text: str
    source: str
    heading_path: str | None = None
    score: float = 0.0
    parent_id: str | None = None
    block_id: str | None = None
    revision: str | None = None


class QueryParam(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=50)
    max_total_tokens: int | None = None
    enable_rerank: bool = True


class RetrievalResult(BaseModel):
    chunks: list[Evidence] = Field(default_factory=list)
