"""模型抽象层：索引与检索只依赖这些 Protocol，不直接依赖厂商 SDK。

切换厂商只需替换适配器与配置，核心逻辑不变。
"""
from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from minirag.schemas import Message

TModel = TypeVar("TModel", bound=BaseModel)


class RerankResult(BaseModel):
    index: int
    score: float


@runtime_checkable
class ChatModel(Protocol):
    """对话模型：强制结构化输出，禁止裸文本回退。"""

    async def generate(
        self,
        messages: list[Message],
        response_model: type[TModel],
        thinking: bool = True,
    ) -> TModel:
        ...


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
