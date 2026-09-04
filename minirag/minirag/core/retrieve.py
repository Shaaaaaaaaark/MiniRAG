"""文本检索编排：Dense + BM25 + RRF → Parent 回填 → Rerank。"""
from __future__ import annotations

import logging

from minirag.config import RetrievalCfg
from minirag.core.ranking import apply_rerank, take_within_budget
from minirag.models.factory import ModelBundle
from minirag.schemas import Evidence, QueryParam, RetrievalResult
from minirag.storage.milvus_store import MilvusStore
from minirag.storage.pg_store import PgStore

_logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, cfg: RetrievalCfg, models: ModelBundle, pg: PgStore, milvus: MilvusStore) -> None:
        self._cfg = cfg
        self._models = models
        self._pg = pg
        self._milvus = milvus

    async def retrieve(self, query: str, param: QueryParam | None = None) -> RetrievalResult:
        param = param or QueryParam()
        top_k = param.top_k or self._cfg.chunk_top_k
        (query_vector,) = await self._models.embedding.embed([query])
        hits = await self._milvus.hybrid_search_chunks(
            query,
            query_vector,
            self._cfg.dense_topk,
            self._cfg.bm25_topk,
            self._cfg.rrf_k,
        )
        chunks = await self._pg.chunks_by_ids([hit.ref_id for hit in hits])
        scores = {hit.ref_id: hit.score for hit in hits}
        for evidence in chunks:
            evidence.score = scores.get(evidence.ref_id, 0.0)
        if param.enable_rerank and chunks:
            chunks = await self._rerank(
                query,
                chunks,
                max(top_k, self._cfg.rerank_topk),
            )
        token_budget = param.max_total_tokens or self._cfg.max_total_tokens
        return RetrievalResult(
            chunks=take_within_budget(chunks, token_budget)[:top_k],
        )

    async def _rerank(self, query: str, chunks: list[Evidence], top_k: int) -> list[Evidence]:
        try:
            results = await self._models.rerank.rerank(
                query,
                [c.text for c in chunks],
                top_k,
            )
        except Exception as err:  # noqa: BLE001 - rerank is an optional enhancement
            _logger.warning("Rerank 调用失败，保留融合排序：%s", err)
            return chunks
        reranked = apply_rerank(chunks, results)
        return reranked or chunks
