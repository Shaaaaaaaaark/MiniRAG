"""Embedding 与 Rerank 模型协议。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class RerankResult(BaseModel):
    index: int
    score: float


@runtime_checkable
class EmbeddingModel(Protocol):
    """向量模型：返回定长稠密向量。"""

    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class RerankModel(Protocol):
    """重排模型：返回 (原始下标, 分数)，按分数降序。"""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[RerankResult]:
        ...
