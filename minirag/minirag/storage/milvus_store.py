"""Milvus Child Chunk 向量存储。

实现说明：
- pymilvus 为同步库，所有网络调用统一用 asyncio.to_thread 包裹，避免阻塞事件循环。
- BM25 由 Milvus 内置 Function 从 text 字段自动生成 sparse 向量（需 Milvus 2.5+）；
  text 字段启用 chinese 分析器以适配中文语料。
- 建表/建库幂等，可安全重复调用。
"""
from __future__ import annotations

import asyncio
from typing import Any

from minirag.config import MilvusCfg
from minirag.schemas import Evidence, make_evidence_id

CHUNK_COLLECTION = "chunk_vectors"

_DENSE_INDEX = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
_SPARSE_INDEX = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}
_TEXT_MAX_LEN = 8192


class MilvusStore:
    def __init__(self, cfg: MilvusCfg, dim: int) -> None:
        self._cfg = cfg
        self._dim = dim
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError("MilvusStore 未连接，请先调用 connect()")
        return self._client

    async def connect(self) -> None:
        """建立连接并确保 Collection 存在。"""
        await asyncio.to_thread(self._connect_sync)

    async def close(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None

    # ------------------------------------------------------------------ #
    # 连接与建表（同步，运行在线程池）
    # ------------------------------------------------------------------ #
    def _connect_sync(self) -> None:
        from pymilvus import MilvusClient

        self._ensure_database()
        client = MilvusClient(uri=self._cfg.uri, db_name=self._cfg.db)
        self._client = client
        self._ensure_chunk_collection(client)
        self._load_collection(CHUNK_COLLECTION)

    def _ensure_database(self) -> None:
        from pymilvus import MilvusClient

        if self._cfg.db in ("", "default"):
            return
        admin = MilvusClient(uri=self._cfg.uri)
        try:
            if self._cfg.db not in admin.list_databases():
                admin.create_database(self._cfg.db)
        finally:
            admin.close()

    def _ensure_chunk_collection(self, client: Any) -> None:
        """chunk_vectors：dense + sparse(BM25 由 text 自动生成)。"""
        from pymilvus import DataType, Function, FunctionType

        if client.has_collection(CHUNK_COLLECTION):
            return

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("heading_path", DataType.VARCHAR, max_length=1024)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=_TEXT_MAX_LEN,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", **_DENSE_INDEX)
        index_params.add_index(field_name="sparse", **_SPARSE_INDEX)
        client.create_collection(CHUNK_COLLECTION, schema=schema, index_params=index_params)

    def _load_collection(self, collection: str) -> None:
        if not self.client.has_collection(collection):
            raise ValueError(f"Milvus collection 不存在: {collection}")
        self.client.load_collection(collection)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    async def upsert_chunk_vectors(
        self,
        ids: list[str],
        dense: list[list[float]],
        texts: list[str],
        document_ids: list[str],
        sources: list[str],
        heading_paths: list[str],
    ) -> None:
        rows = [
            {
                "id": ids[i],
                "document_id": document_ids[i],
                "source": sources[i],
                "heading_path": heading_paths[i],
                "text": texts[i],
                "dense": dense[i],
            }
            for i in range(len(ids))
        ]
        await self._upsert(CHUNK_COLLECTION, rows)

    async def _upsert(self, collection: str, rows: list[dict]) -> None:
        if not rows:
            return
        await asyncio.to_thread(self._load_collection, collection)
        await asyncio.to_thread(self.client.upsert, collection_name=collection, data=rows)

    async def delete_chunk_vectors(self, ids: list[str]) -> None:
        await self._delete(CHUNK_COLLECTION, ids)

    async def _delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        await asyncio.to_thread(self._load_collection, collection)
        await asyncio.to_thread(self.client.delete, collection_name=collection, pks=ids)

    async def flush(self) -> None:
        """Make all completed index writes visible before reporting success."""
        await asyncio.to_thread(self.client.flush, collection_name=CHUNK_COLLECTION)

    async def hybrid_search_chunks(
        self,
        query_text: str,
        query_vec: list[float],
        dense_k: int,
        bm25_k: int,
        rrf_k: int = 60,
    ) -> list[Evidence]:
        """dense + sparse(BM25) 混合检索，RRFRanker 融合。"""
        return await asyncio.to_thread(
            self._hybrid_search_sync,
            query_text,
            query_vec,
            dense_k,
            bm25_k,
            rrf_k,
        )

    def _hybrid_search_sync(
        self,
        query_text: str,
        query_vec: list[float],
        dense_k: int,
        bm25_k: int,
        rrf_k: int,
    ) -> list[Evidence]:
        from pymilvus import AnnSearchRequest, RRFRanker

        self._load_collection(CHUNK_COLLECTION)
        dense_req = AnnSearchRequest(
            data=[query_vec],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=dense_k,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=bm25_k,
        )
        hits = self.client.hybrid_search(
            collection_name=CHUNK_COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(rrf_k),
            limit=max(dense_k, bm25_k),
            output_fields=["id", "text", "source", "heading_path"],
        )
        return self._to_chunk_evidences(hits)

    # ------------------------------------------------------------------ #
    # 结果映射
    # ------------------------------------------------------------------ #
    def _to_chunk_evidences(self, hits: Any) -> list[Evidence]:
        out: list[Evidence] = []
        for hit in self._first_group(hits):
            entity = hit.get("entity", {})
            out.append(
                Evidence(
                    evidence_id=make_evidence_id(entity.get("id", "")),
                    ref_id=entity.get("id", ""),
                    text=entity.get("text", ""),
                    source=entity.get("source", ""),
                    heading_path=entity.get("heading_path"),
                    score=float(hit.get("distance", 0.0)),
                )
            )
        return out

    @staticmethod
    def _first_group(hits: Any) -> list[dict]:
        if not hits:
            return []
        return list(hits[0])
