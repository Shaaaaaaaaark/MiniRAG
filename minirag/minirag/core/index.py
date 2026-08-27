"""索引编排：解析 → 切分 → 抽取 → 融合 → 向量化 → 落库。

幂等：同 document_id 重新索引先删旧 chunk 再写（首版不做增量 diff）。
图节点/边为全局融合（跨文档共享），按稳定 ID upsert。
"""
from __future__ import annotations

import asyncio
import logging

from minirag.config import Settings
from minirag.core import extract as extract_mod
from minirag.core.chunker import (
    HeaderTokenChunker,
    ParentChildChunker,
    document_id,
    parse_document,
)
from minirag.core.fusion import (
    canonical_edge_ids,
    entity_id,
    merge_entity,
    merge_relation,
    relation_id,
)
from minirag.core.tokenizer import count_tokens
from minirag.models.factory import ModelBundle
from minirag.schemas import (
    Chunk,
    DocumentInput,
    Entity,
    ExtractedGraph,
    IndexReport,
    Relation,
)
from minirag.storage.milvus_store import MilvusStore
from minirag.storage.pg_store import PgStore

_logger = logging.getLogger(__name__)
_EXTRACT_CONCURRENCY = 4


class Indexer:
    def __init__(self, settings: Settings, models: ModelBundle, pg: PgStore, milvus: MilvusStore) -> None:
        self._settings = settings
        self._models = models
        self._pg = pg
        self._milvus = milvus
        self._chunkers = {
            "header_token": HeaderTokenChunker(),
            "parent_child": ParentChildChunker(),
        }

    async def index_document(self, doc_input: DocumentInput, gleaning: bool = False) -> IndexReport:
        doc = parse_document(doc_input)

        old_document_ids = await self._pg.document_ids_for_source(doc.source)
        for old_document_id in {doc.document_id, *old_document_ids}:
            await self._delete_document_by_id(old_document_id)
        await self._pg.upsert_document(doc, status="parsing")

        try:
            strategy = doc_input.chunking_strategy or self._settings.chunker
            try:
                chunker = self._chunkers[strategy]
            except KeyError as err:
                raise ValueError(f"不支持的 chunking strategy: {strategy}") from err
            chunks = chunker.split(doc)
            await self._pg.insert_chunks(chunks)

            entities: dict[str, Entity] = {}
            relations: dict[str, Relation] = {}
            extraction_chunks = (
                _graph_extraction_chunks(chunks)
                if self._settings.graph_enabled
                else []
            )
            semaphore = asyncio.Semaphore(_EXTRACT_CONCURRENCY)

            async def extract_one(
                chunk: Chunk,
            ) -> tuple[Chunk, ExtractedGraph | None]:
                async with semaphore:
                    try:
                        graph = await extract_mod.extract(
                            chunk,
                            self._models.chat,
                            gleaning=gleaning,
                        )
                        return chunk, graph
                    except Exception as err:  # noqa: BLE001 - one failed graph must not abort text indexing
                        _logger.warning(
                            "chunk %s 图谱抽取失败，已跳过：%s",
                            chunk.id,
                            err,
                        )
                        return chunk, None

            extraction_results = await asyncio.gather(
                *(extract_one(chunk) for chunk in extraction_chunks)
            )
            skipped = 0
            for chunk, graph in extraction_results:
                if graph is None:
                    skipped += 1
                    continue
                await self._merge_graph(chunk, graph, entities, relations)

            await self._embed_and_store(doc.source, chunks, entities, relations)
            await self._milvus.flush()
            await self._pg.upsert_document(doc, status="done")
        except Exception:
            await self._delete_document_by_id(doc.document_id)
            await self._milvus.flush()
            raise

        if skipped:
            _logger.warning(
                "文档 %s 索引完成，%d/%d 个父块图谱抽取被跳过",
                doc.document_id,
                skipped,
                len(extraction_chunks),
            )

        return IndexReport(
            document_id=doc.document_id,
            chunks=len(chunks),
            entities=len(entities),
            relations=len(relations),
        )

    async def delete_document(self, source: str, source_id: str | None = None) -> None:
        target_id = document_id(source, source_id)
        document_ids = {target_id, *await self._pg.document_ids_for_source(source)}
        for doc_id in document_ids:
            await self._delete_document_by_id(doc_id)
        await self._milvus.flush()

    async def _delete_document_by_id(self, doc_id: str) -> None:
        old_chunk_ids = await self._pg.chunk_ids_for_document(doc_id)
        if old_chunk_ids:
            await self._milvus.delete_chunk_vectors(old_chunk_ids)
            deleted_node_ids, deleted_edge_ids = await self._pg.detach_source_chunks_from_graph(
                old_chunk_ids
            )
            await self._milvus.delete_entity_vectors(deleted_node_ids)
            await self._milvus.delete_relation_vectors(deleted_edge_ids)
        await self._pg.delete_document(doc_id)

    async def _merge_graph(
        self,
        chunk: Chunk,
        graph: ExtractedGraph,
        entities: dict[str, Entity],
        relations: dict[str, Relation],
    ) -> None:
        """Deterministically merge one extracted parent into PostgreSQL."""
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


def _graph_extraction_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Use one complete parent per graph extraction; keep children for retrieval."""
    output: list[Chunk] = []
    seen_parents: set[str] = set()
    for chunk in chunks:
        if not chunk.parent_id:
            output.append(chunk)
            continue
        if chunk.parent_id in seen_parents:
            continue
        seen_parents.add(chunk.parent_id)
        parent_content = chunk.parent_content or chunk.content
        output.append(
            chunk.model_copy(
                update={
                    "content": parent_content,
                    "token_count": count_tokens(parent_content),
                }
            )
        )
    return output
