"""PostgreSQL 存储：文档/分块元数据 + 知识图谱（普通表实现，无需 AGE 扩展）。

图模型（借鉴官方 pgtable 实现的关键点）：
- graph_nodes：实体节点，id = ent_<sha1>，与 Milvus entity 向量 id 一致。
- graph_edges：关系边，id = rel_<sha1>，与 Milvus relation 向量 id 一致；
  边按规范化顺序存储 src_id = min(a,b)、tgt_id = max(a,b)，用 Python min/max
  （不用 SQL LEAST/GREATEST，避免非 ASCII 中文在非 C collation 下排序分叉导致重复边）。

检索只需「一跳邻接 + 度数」，全部普通索引 SQL。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from minirag.config import PostgresCfg
from minirag.schemas import Chunk, Entity, Evidence, ParsedDocument, Relation, make_evidence_id

if TYPE_CHECKING:
    from asyncpg import Pool

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    title        TEXT,
    revision     TEXT,
    parse_status TEXT NOT NULL DEFAULT 'pending',
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord          INT  NOT NULL,
    heading_path TEXT,
    content      TEXT NOT NULL,
    token_count  INT  NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    type          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    source_chunks TEXT[] NOT NULL DEFAULT '{}',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_name ON graph_nodes(name);

CREATE TABLE IF NOT EXISTS graph_edges (
    id            TEXT PRIMARY KEY,
    src_id        TEXT NOT NULL,
    tgt_id        TEXT NOT NULL,
    keywords      TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    weight        REAL NOT NULL DEFAULT 1.0,
    source_chunks TEXT[] NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_tgt ON graph_edges(tgt_id);
"""


class PgStore:
    """PostgreSQL 数据访问层。连接池在 startup 时通过 connect() 建立。"""

    def __init__(self, cfg: PostgresCfg) -> None:
        self._cfg = cfg
        self._pool: "Pool | None" = None

    @property
    def pool(self) -> "Pool":
        if self._pool is None:
            raise RuntimeError("PgStore 未连接，请先调用 connect()")
        return self._pool

    async def connect(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(dsn=self._cfg.dsn)
        async with self.pool.acquire() as conn:
            await conn.execute(DDL)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    # ---- documents ----
    async def upsert_document(self, doc: ParsedDocument, status: str = "done") -> None:
        sql = """
        INSERT INTO documents(id, source, title, revision, parse_status, updated_at)
        VALUES($1,$2,$3,$4,$5, now())
        ON CONFLICT(id) DO UPDATE SET
            source=EXCLUDED.source, title=EXCLUDED.title,
            revision=EXCLUDED.revision, parse_status=EXCLUDED.parse_status,
            updated_at=now();
        """
        async with self.pool.acquire() as conn:
            await conn.execute(sql, doc.document_id, doc.source, doc.title, doc.revision, status)

    async def delete_document_children(self, document_id: str) -> None:
        """重新索引前清理旧 chunk（级联）。图节点/边为全局融合，单独处理。"""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE document_id=$1", document_id)

    # ---- chunks ----
    async def insert_chunks(self, chunks: list[Chunk]) -> None:
        sql = """
        INSERT INTO chunks(id, document_id, ord, heading_path, content, token_count)
        VALUES($1,$2,$3,$4,$5,$6)
        ON CONFLICT(id) DO UPDATE SET
            content=EXCLUDED.content, heading_path=EXCLUDED.heading_path,
            token_count=EXCLUDED.token_count;
        """
        rows = [(c.id, c.document_id, c.ord, c.heading_path, c.content, c.token_count) for c in chunks]
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, rows)

    async def chunks_by_ids(self, chunk_ids: list[str]) -> list[Evidence]:
        """按 chunk id 批量取正文，转为 Evidence（图扩展用）。"""
        if not chunk_ids:
            return []
        sql = """
        SELECT c.id, c.content, c.heading_path, d.source
        FROM chunks c JOIN documents d ON c.document_id = d.id
        WHERE c.id = ANY($1::text[])
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, chunk_ids)
        return [
            Evidence(
                evidence_id=make_evidence_id("chunk", r["id"]),
                kind="chunk",
                ref_id=r["id"],
                text=r["content"],
                source=r["source"],
                heading_path=r["heading_path"],
            )
            for r in rows
        ]

    # ---- graph: 写入 ----
    async def upsert_node(self, node_id: str, entity: Entity, source_chunk: str) -> None:
        sql = """
        INSERT INTO graph_nodes(id, name, type, description, source_chunks, updated_at)
        VALUES($1,$2,$3,$4, ARRAY[$5], now())
        ON CONFLICT(id) DO UPDATE SET
            description=EXCLUDED.description,
            source_chunks=array(SELECT DISTINCT unnest(graph_nodes.source_chunks || EXCLUDED.source_chunks)),
            updated_at=now();
        """
        async with self.pool.acquire() as conn:
            await conn.execute(sql, node_id, entity.name, entity.type, entity.description, source_chunk)

    async def upsert_edge(self, edge_id: str, src_id: str, tgt_id: str, rel: Relation, source_chunk: str) -> None:
        """边按规范化顺序存储（src_id <= tgt_id，由调用方保证）。"""
        sql = """
        INSERT INTO graph_edges(id, src_id, tgt_id, keywords, description, weight, source_chunks)
        VALUES($1,$2,$3,$4,$5,$6, ARRAY[$7])
        ON CONFLICT(id) DO UPDATE SET
            keywords=EXCLUDED.keywords, description=EXCLUDED.description,
            weight=graph_edges.weight + EXCLUDED.weight,
            source_chunks=array(SELECT DISTINCT unnest(graph_edges.source_chunks || EXCLUDED.source_chunks));
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                sql, edge_id, src_id, tgt_id, rel.keywords, rel.description, rel.weight, source_chunk
            )

    # ---- graph: 检索（一跳邻接 + 度数） ----
    async def node_degree(self, node_id: str) -> int:
        sql = "SELECT count(*) FROM graph_edges WHERE src_id=$1 OR tgt_id=$1"
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval(sql, node_id) or 0)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        sql = """
        SELECT nid, count(*) AS deg FROM (
            SELECT src_id AS nid FROM graph_edges WHERE src_id = ANY($1::text[])
            UNION ALL
            SELECT tgt_id AS nid FROM graph_edges WHERE tgt_id = ANY($1::text[])
        ) t GROUP BY nid
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, node_ids)
        return {r["nid"]: int(r["deg"]) for r in rows}

    async def get_node_edges(self, node_id: str) -> list[dict]:
        """取某节点的直接邻边（一跳）。返回边属性 + 对端节点 id。"""
        sql = """
        SELECT id, src_id, tgt_id, keywords, description, weight, source_chunks
        FROM graph_edges WHERE src_id=$1 OR tgt_id=$1
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, node_id)
        return [dict(r) for r in rows]

    async def edges_by_ids(self, edge_ids: list[str]) -> list[dict]:
        if not edge_ids:
            return []
        sql = """
        SELECT id, src_id, tgt_id, keywords, description, weight, source_chunks
        FROM graph_edges WHERE id = ANY($1::text[])
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, edge_ids)
        return [dict(r) for r in rows]

    async def nodes_by_ids(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        sql = """
        SELECT id, name, type, description, source_chunks
        FROM graph_nodes WHERE id = ANY($1::text[])
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, node_ids)
        return {r["id"]: dict(r) for r in rows}
