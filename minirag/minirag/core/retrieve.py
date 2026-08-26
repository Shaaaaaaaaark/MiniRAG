"""检索编排（LightRAG 内核）：

双层关键词抽取 → 按模式召回(entity/relation/chunk) → chunk rerank
→ 分层 token 预算截断(实体/关系/chunk 各限额) → 只返回结构化证据。
"""
from __future__ import annotations

from minirag.config import RetrievalCfg
from minirag.core import keywords as kw_mod
from minirag.core.fusion_retrieval import apply_rerank, take_within_budget
from minirag.core.modes import perform_search
from minirag.models.factory import ModelBundle
from minirag.schemas import Evidence, QueryParam, RetrievalResult
from minirag.storage.milvus_store import MilvusStore
from minirag.storage.pg_store import PgStore


class Retriever:
    def __init__(self, cfg: RetrievalCfg, models: ModelBundle, pg: PgStore, milvus: MilvusStore) -> None:
        self._cfg = cfg
        self._models = models
        self._pg = pg
        self._milvus = milvus

    async def retrieve(self, query: str, param: QueryParam | None = None) -> RetrievalResult:
        param = param or QueryParam()
        cfg = self._effective_cfg(param)

        # 双层关键词（naive 模式也抽，但不使用，成本可忽略；如需极致可跳过）
        kw = await kw_mod.extract_keywords(query, self._models.chat)

        recall = await perform_search(
            param.mode, query, kw, cfg, self._models, self._milvus, self._pg
        )

        # chunk 走 rerank（实体/关系已由图 rank 排序）
        chunks = recall.chunks
        if param.enable_rerank and chunks:
            chunks = await self._rerank(query, chunks, cfg.rerank_topk)

        # 分层 token 预算截断
        entities = take_within_budget(recall.entities, cfg.max_entity_tokens)
        relationships = take_within_budget(recall.relationships, cfg.max_relation_tokens)
        # chunk 预算 = 总预算扣除实体/关系已用（简化官方动态余量）
        used = sum(len(e.text) for e in entities) + sum(len(r.text) for r in relationships)
        chunk_budget = max(cfg.max_total_tokens - used // 4, cfg.max_total_tokens // 3)
        chunks = take_within_budget(chunks, chunk_budget)
        if param.chunk_top_k is not None:
            chunks = chunks[: param.chunk_top_k]

        return RetrievalResult(
            keywords=kw,
            entities=entities,
            relationships=relationships,
            chunks=chunks,
        )

    def _effective_cfg(self, param: QueryParam) -> RetrievalCfg:
        """用 QueryParam 覆盖 config 默认值，得到本次生效参数。"""
        data = self._cfg.model_dump()
        if param.top_k is not None:
            data["entity_topk"] = param.top_k
            data["relation_topk"] = param.top_k
        if param.max_entity_tokens is not None:
            data["max_entity_tokens"] = param.max_entity_tokens
        if param.max_relation_tokens is not None:
            data["max_relation_tokens"] = param.max_relation_tokens
        if param.max_total_tokens is not None:
            data["max_total_tokens"] = param.max_total_tokens
        return RetrievalCfg.model_validate(data)

    async def _rerank(self, query: str, chunks: list[Evidence], top_k: int) -> list[Evidence]:
        results = await self._models.rerank.rerank(query, [c.text for c in chunks], top_k)
        reranked = apply_rerank(chunks, results)
        return reranked or chunks
