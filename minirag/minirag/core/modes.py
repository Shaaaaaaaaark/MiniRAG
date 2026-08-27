"""五种检索模式路由（对应官方 _perform_kg_search 精简版）。

各模式产出「实体证据 / 关系证据 / chunk 证据」三组原始召回，
后续融合/rerank/截断在 retrieve.py 统一处理。

- local ：low 关键词 → 实体向量 → 图取邻边(关系) + 实体来源 chunk
- global：high 关键词 → 关系向量 → 取两端实体 + 关系来源 chunk
- hybrid：local + global，round-robin 合并
- mix   ：hybrid + query 向量召回 chunk（默认）
- text  ：Dense + BM25 召回 chunk（默认）
- naive ：text 的兼容别名
"""
from __future__ import annotations

from dataclasses import dataclass, field

from minirag.config import RetrievalCfg
from minirag.core import graph as graph_mod
from minirag.core.fusion_retrieval import round_robin_merge
from minirag.models.factory import ModelBundle
from minirag.schemas import Evidence, Keywords, QueryMode
from minirag.storage.milvus_store import MilvusStore
from minirag.storage.pg_store import PgStore


@dataclass
class RawRecall:
    entities: list[Evidence] = field(default_factory=list)
    relationships: list[Evidence] = field(default_factory=list)
    chunks: list[Evidence] = field(default_factory=list)


async def _local_recall(
    kw: Keywords, cfg: RetrievalCfg, models: ModelBundle, milvus: MilvusStore, pg: PgStore
) -> RawRecall:
    """low 关键词 → 实体向量 → 一跳邻边(关系) + 实体来源 chunk。"""
    low_text = " ".join(kw.low_level)
    if not low_text:
        return RawRecall()
    (low_vec,) = await models.embedding.embed([low_text])
    entities = await milvus.search_entities(low_vec, cfg.entity_topk)
    relations = await graph_mod.expand_entities_to_relations(entities, pg)
    chunks = await graph_mod.source_chunks_of_entities(entities, pg)
    return RawRecall(entities=entities, relationships=relations, chunks=chunks)


async def _global_recall(
    kw: Keywords, cfg: RetrievalCfg, models: ModelBundle, milvus: MilvusStore, pg: PgStore
) -> RawRecall:
    """high 关键词 → 关系向量 → 两端实体 + 关系来源 chunk。"""
    high_text = " ".join(kw.high_level)
    if not high_text:
        return RawRecall()
    (high_vec,) = await models.embedding.embed([high_text])
    relations = await milvus.search_relations(high_vec, cfg.relation_topk)
    entities = await graph_mod.expand_relations_to_entities(relations, pg)
    chunks = await graph_mod.source_chunks_of_edges(relations, pg)
    return RawRecall(entities=entities, relationships=relations, chunks=chunks)


async def _chunk_recall(
    query: str,
    cfg: RetrievalCfg,
    models: ModelBundle,
    milvus: MilvusStore,
    pg: PgStore,
) -> list[Evidence]:
    """query → chunk 向量 + BM25 混合召回。"""
    (query_vec,) = await models.embedding.embed([query])
    hits = await milvus.hybrid_search_chunks(query, query_vec, cfg.dense_topk, cfg.bm25_topk)
    hydrated = await pg.chunks_by_ids([hit.ref_id for hit in hits])
    scores = {hit.ref_id: hit.score for hit in hits}
    for evidence in hydrated:
        evidence.score = scores.get(evidence.ref_id, 0.0)
    return hydrated


async def perform_search(
    mode: QueryMode,
    query: str,
    kw: Keywords,
    cfg: RetrievalCfg,
    models: ModelBundle,
    milvus: MilvusStore,
    pg: PgStore,
) -> RawRecall:
    """按模式路由，返回三组原始召回证据。"""
    if mode in {"text", "naive"}:
        chunks = await _chunk_recall(query, cfg, models, milvus, pg)
        return RawRecall(chunks=chunks)

    if mode == "local":
        return await _local_recall(kw, cfg, models, milvus, pg)

    if mode == "global":
        return await _global_recall(kw, cfg, models, milvus, pg)

    # hybrid / mix
    local = await _local_recall(kw, cfg, models, milvus, pg)
    glob = await _global_recall(kw, cfg, models, milvus, pg)
    merged = RawRecall(
        entities=round_robin_merge(local.entities, glob.entities),
        relationships=round_robin_merge(local.relationships, glob.relationships),
        chunks=round_robin_merge(local.chunks, glob.chunks),
    )
    if mode == "mix":
        vector_chunks = await _chunk_recall(query, cfg, models, milvus, pg)
        merged.chunks = round_robin_merge(merged.chunks, vector_chunks)
    return merged
