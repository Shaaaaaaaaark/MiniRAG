"""百炼文本重排适配器（DashScope 原生 rerank 接口）。

config 里 rerank.base_url 与 embedding 一致，此处自动剥离兼容层尾巴取服务根，
再拼接原生 rerank 路径：
  POST {root}/api/v1/services/rerank/text-rerank/text-rerank
"""
from __future__ import annotations

import asyncio

import httpx

from minirag.config import ModelCfg
from minirag.models.base import RerankResult

_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


class BailianRerankModel:
    def __init__(self, cfg: ModelCfg) -> None:
        self._cfg = cfg
        self._url = self._build_url(cfg.base_url)
        self._headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_url(base_url: str) -> str:
        root = base_url
        for suffix in ("/compatible-mode/v1", "/compatible-api/v1", "/v1"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
                break
        return root.rstrip("/") + _RERANK_PATH

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[RerankResult]:
        if not documents:
            return []
        body = {
            "model": self._cfg.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_k, "return_documents": False},
        }
        data = await self._post(body)
        results = data.get("output", {}).get("results", [])
        return [
            RerankResult(index=r["index"], score=float(r.get("relevance_score", 0.0)))
            for r in results
        ]

    async def _post(self, body: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._cfg.effective_timeout) as client:
                    resp = await client.post(self._url, json=body, headers=self._headers)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as err:  # noqa: BLE001 - provider errors are retried uniformly
                last_err = err
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"BailianRerankModel.rerank 失败: {last_err}")
