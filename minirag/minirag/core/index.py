"""索引编排：解析 → 切分 → 抽取 → 融合 → 向量化 → 落库。

幂等：同 document_id 重新索引先删旧 chunk 再写（首版不做增量 diff）。
图节点/边为全局融合（跨文档共享），按稳定 ID upsert。
"""
from __future__ import annotations

import logging

from minirag.config import Settings
from minirag.core import extract as extract_mod
from minirag.core.chunker import HeaderTokenChunker, parse_document
from minirag.core.fusion import (
    canonical_edge_ids,
    entity_id,
    merge_entity,
    merge_relation,
    relation_id,
)
from minirag.models.factory import ModelBundle
from minirag.storage.milvus_store import MilvusStore
from minirag.storage.pg_store import PgStore
from minirag.schemas import Chunk, DocumentInput, Entity, IndexReport, Relation

_logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, settings: Settings, models: ModelBundle, pg: PgStore, milvus: MilvusStore) -> None:
        self._settings = settings
        self._models = models
        self._pg = pg
        self._milvus = milvus
        self._chunker = HeaderTokenChunker()

    async def index_document(self, doc_input: DocumentInput, gleaning: bool = False) -> IndexReport:
        doc = parse_document(doc_input)

        await self._pg.delete_document_children(doc.document_id)
        await self._pg.upsert_document(doc, status="parsing")

        chunks = self._chunker.split(doc)
        await self._pg.insert_chunks(chunks)

        entities: dict[str, Entity] = {}
        relations: dict[str, Relation] = {}
        skipped = 0
        for chunk in chunks:
            try:
                await self._extract_and_merge(chunk, entities, relations, gleaning)
            except Exception as err:
                # 单块图谱抽取失败（多为 LLM 结构化输出抖动）不阻断整篇索引：
                # chunk 向量是主检索路径，实体/关系为增强层，跳过并记录即可。
                skipped += 1
                _logger.warning("chunk %s 图谱抽取失败，已跳过：%s", chunk.id, err)

        await self._embed_and_store(doc.source, chunks, entities, relations)
        await self._pg.upsert_document(doc, status="done")

        if skipped:
            _logger.warning("文档 %s 索引完成，%d/%d 个 chunk 图谱抽取被跳过", doc.document_id, skipped, len(chunks))

        return IndexReport(
            document_id=doc.document_id,
            chunks=len(chunks),
            entities=len(entities),
            relations=len(relations),
        )

    async def _extract_and_merge(
        self, chunk: Chunk, entities: dict[str, Entity], relations: dict[str, Relation], gleaning: bool
    ) -> None:
        """抽取单个 chunk 的图谱并融合进全局字典与 PG。"""
        graph = await extract_mod.extract(chunk, self._models.chat, gleaning=gleaning)
        for ent in graph.entities:
            eid = entity_id(ent.name)
            merged = merge_entity(entities.get(eid), ent)
            entities[eid] = merged
            await self._pg.upsert_node(eid, merged, chunk.id)
        for rel in graph.relations:
            rid = relation_id(rel.src, rel.dst)
            merged_rel = merge_relation(relations.get(rid), rel)
            relations[rid] = merged_rel
            src_id, tgt_id = canonical_edge_ids(rel.src, rel.dst)
            await self._pg.upsert_edge(rid, src_id, tgt_id, merged_rel, chunk.id)

    async def _embed_and_store(
        self,
        source: str,
        chunks: list[Chunk],
        entities: dict[str, Entity],
        relations: dict[str, Relation],
    ) -> None:
        """对 chunk / 实体 / 关系分别向量化并写入 Milvus。"""
        embed = self._models.embedding
        if chunks:
            chunk_vecs = await embed.embed([c.content for c in chunks])
            await self._milvus.upsert_chunk_vectors(
                ids=[c.id for c in chunks],
                dense=chunk_vecs,
                texts=[c.content for c in chunks],
                document_ids=[c.document_id for c in chunks],
                sources=[source for _ in chunks],
                heading_paths=[c.heading_path for c in chunks],
            )
        if entities:
            ids = list(entities.keys())
            texts = [f"{entities[i].name}: {entities[i].description}" for i in ids]
            ent_vecs = await embed.embed(texts)
            await self._milvus.upsert_entity_vectors(ids, ent_vecs, texts)
        if relations:
            ids = list(relations.keys())
            texts = [f"{relations[i].keywords}: {relations[i].description}" for i in ids]
            rel_vecs = await embed.embed(texts)
            await self._milvus.upsert_relation_vectors(ids, rel_vecs, texts)
