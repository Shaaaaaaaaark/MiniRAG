"""百炼 text-embedding-v4 向量模型适配器（OpenAI 兼容）。

单批 ≤ 10 条（百炼服务端限制），返回定长向量（默认 1024 维）。
"""
from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from minirag.config import ModelCfg

_MAX_BATCH = 10


class BailianEmbeddingModel:
    def __init__(self, cfg: ModelCfg) -> None:
        self._cfg = cfg
        self.dimensions = cfg.dimensions or 1024
        self._client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.effective_timeout,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            batch = texts[start : start + _MAX_BATCH]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                resp = await self._client.embeddings.create(
                    model=self._cfg.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
                return [item.embedding for item in resp.data]
            except Exception as err:
                last_err = err
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"BailianEmbeddingModel.embed 失败: {last_err}")
