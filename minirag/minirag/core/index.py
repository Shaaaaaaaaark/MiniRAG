"""索引编排：解析 → Parent-Child 切分 → 向量化 → 落库。"""
from __future__ import annotations

from minirag.core.chunker import ParentChildChunker, document_id, parse_document
from minirag.models.factory import ModelBundle
from minirag.schemas import Chunk, DocumentInput, IndexReport
from minirag.storage.milvus_store import MilvusStore
from minirag.storage.pg_store import PgStore


class Indexer:
    def __init__(self, models: ModelBundle, pg: PgStore, milvus: MilvusStore) -> None:
        self._models = models
        self._pg = pg
        self._milvus = milvus
        self._chunker = ParentChildChunker()

    async def index_document(self, doc_input: DocumentInput) -> IndexReport:
        doc = parse_document(doc_input)

        old_document_ids = await self._pg.document_ids_for_source(doc.source)
        for old_document_id in {doc.document_id, *old_document_ids}:
            await self._delete_document_by_id(old_document_id)
        await self._pg.upsert_document(doc, status="parsing")

        try:
            chunks = self._chunker.split(doc)
            await self._pg.insert_chunks(chunks)
            await self._embed_and_store(doc.source, chunks)
            await self._milvus.flush()
            await self._pg.upsert_document(doc, status="done")
        except Exception:
            await self._delete_document_by_id(doc.document_id)
            await self._milvus.flush()
            raise

        return IndexReport(document_id=doc.document_id, chunks=len(chunks))

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
        await self._pg.delete_document(doc_id)

    async def _embed_and_store(self, source: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await self._models.embedding.embed([chunk.content for chunk in chunks])
        await self._milvus.upsert_chunk_vectors(
            ids=[chunk.id for chunk in chunks],
            dense=vectors,
            texts=[chunk.content for chunk in chunks],
            document_ids=[chunk.document_id for chunk in chunks],
            sources=[source] * len(chunks),
            heading_paths=[chunk.heading_path for chunk in chunks],
        )
