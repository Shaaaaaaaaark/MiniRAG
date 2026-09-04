"""DashScope-compatible reranker (RRF fallback is handled by the caller)."""
from __future__ import annotations

import asyncio

import httpx
from langchain_core.documents import Document

from minirag_langchain.config import RerankConfig

_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


def _rerank_url(base_url: str) -> str:
    """Strip the OpenAI-compatible suffix to reach the native rerank endpoint."""
    root = base_url
    for suffix in ("/compatible-mode/v1", "/compatible-api/v1", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root.rstrip("/") + _RERANK_PATH


async def rerank(
    config: RerankConfig,
    query: str,
    documents: list[Document],
    top_k: int,
) -> list[tuple[Document, float]]:
    if not documents:
        return []
    url = _rerank_url(config.base_url)
    body = {
        "model": config.model,
        "input": {
            "query": query,
            "documents": [document.page_content for document in documents],
        },
        "parameters": {"top_n": min(top_k, len(documents)), "return_documents": False},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
            results = response.json().get("output", {}).get("results", [])
            return [
                (documents[item["index"]], float(item["relevance_score"]))
                for item in results
                if 0 <= int(item["index"]) < len(documents)
            ]
        except Exception as error:  # noqa: BLE001 - optional stage retries uniformly
            last_error = error
            if attempt < config.max_retries:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"rerank failed: {last_error}")
