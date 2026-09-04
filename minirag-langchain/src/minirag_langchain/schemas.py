"""Public request and evidence contracts for the LangChain baseline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["text"] = "text"
    top_k: int | None = Field(default=None, ge=1, le=50)
    chunk_top_k: int | None = Field(default=None, ge=1, le=50)
    enable_rerank: bool = True

    @property
    def effective_top_k(self) -> int:
        return self.chunk_top_k or self.top_k or 8


class Evidence(BaseModel):
    evidence_id: str
    ref_id: str
    text: str
    source: str
    heading_path: str = ""
    score: float = 0.0
    parent_id: str
    block_id: str = ""
    revision: str = ""


class RetrieveResponse(BaseModel):
    chunks: list[Evidence] = Field(default_factory=list)
    count: int = 0
