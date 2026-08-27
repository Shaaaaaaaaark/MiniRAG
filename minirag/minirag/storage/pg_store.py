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
from minirag.schemas import (
    Chunk,
    Entity,
    Evidence,
    ParsedDocument,
    Relation,
    make_evidence_id,
)

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
    token_count  INT  NOT NULL,
    parent_id    TEXT,
    parent_content TEXT,
    block_id     TEXT
);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_content TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS block_id TEXT;
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);

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
        self._pool: Pool | None = None

    @property
    def pool(self) -> Pool:
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

    async def chunk_ids_for_document(self, document_id: str) -> list[str]:
        sql = "SELECT id FROM chunks WHERE document_id=$1 ORDER BY ord"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, document_id)
        return [r["id"] for r in rows]

    async def document_ids_for_source(self, source: str) -> list[str]:
        sql = "SELECT id FROM documents WHERE source=$1"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, source)
        return [r["id"] for r in rows]

    async def delete_document(self, document_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM documents WHERE id=$1", document_id)

    async def detach_source_chunks_from_graph(self, chunk_ids: list[str]) -> tuple[list[str], list[str]]:
        """从图谱节点/边里移除指定 chunk 来源，并删除失去来源的图谱行。

        返回值为 (deleted_node_ids, deleted_edge_ids)，调用方据此清理对应向量。
        """
        if not chunk_ids:
            return [], []

        update_edges_sql = """
        UPDATE graph_edges AS ge
        SET source_chunks = ARRAY(
            SELECT DISTINCT chunk_id
            FROM unnest(ge.source_chunks) AS source_chunk(chunk_id)
            WHERE chunk_id <> ALL($1::text[])
        )
        WHERE ge.source_chunks && $1::text[]
        """
        delete_empty_edges_sql = """
        DELETE FROM graph_edges
        WHERE cardinality(source_chunks) = 0
        RETURNING id
        """
        update_nodes_sql = """
        UPDATE graph_nodes AS gn
        SET source_chunks = ARRAY(
            SELECT DISTINCT chunk_id
            FROM unnest(gn.source_chunks) AS source_chunk(chunk_id)
            WHERE chunk_id <> ALL($1::text[])
        ),
            updated_at = now()
        WHERE gn.source_chunks && $1::text[]
        """
        delete_empty_nodes_sql = """
        DELETE FROM graph_nodes
        WHERE cardinality(source_chunks) = 0
        RETURNING id
        """
        delete_orphan_edges_sql = """
        DELETE FROM graph_edges AS ge
        WHERE NOT EXISTS (SELECT 1 FROM graph_nodes AS gn WHERE gn.id = ge.src_id)
           OR NOT EXISTS (SELECT 1 FROM graph_nodes AS gn WHERE gn.id = ge.tgt_id)
        RETURNING id
        """

        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(update_edges_sql, chunk_ids)
            empty_edge_rows = await conn.fetch(delete_empty_edges_sql)
            await conn.execute(update_nodes_sql, chunk_ids)
            empty_node_rows = await conn.fetch(delete_empty_nodes_sql)
            orphan_edge_rows = await conn.fetch(delete_orphan_edges_sql)

        deleted_node_ids = [r["id"] for r in empty_node_rows]
        deleted_edge_ids = [r["id"] for r in [*empty_edge_rows, *orphan_edge_rows]]
        return list(dict.fromkeys(deleted_node_ids)), list(dict.fromkeys(deleted_edge_ids))

    async def prune_orphan_graph_sources(self) -> tuple[list[str], list[str]]:
        """Remove graph contributions whose source chunks no longer exist."""
        prune_edges_sql = """
        UPDATE graph_edges AS ge
        SET source_chunks = ARRAY(
            SELECT source_chunk
            FROM unnest(ge.source_chunks) AS source(source_chunk)
            WHERE EXISTS (SELECT 1 FROM chunks AS c WHERE c.id = source_chunk)
        )
        WHERE EXISTS (
            SELECT 1
            FROM unnest(ge.source_chunks) AS source(source_chunk)
            WHERE NOT EXISTS (SELECT 1 FROM chunks AS c WHERE c.id = source_chunk)
        )
        """
        prune_nodes_sql = """
        UPDATE graph_nodes AS gn
        SET source_chunks = ARRAY(
            SELECT source_chunk
            FROM unnest(gn.source_chunks) AS source(source_chunk)
            WHERE EXISTS (SELECT 1 FROM chunks AS c WHERE c.id = source_chunk)
        ),
            updated_at = now()
        WHERE EXISTS (
            SELECT 1
            FROM unnest(gn.source_chunks) AS source(source_chunk)
            WHERE NOT EXISTS (SELECT 1 FROM chunks AS c WHERE c.id = source_chunk)
        )
        """
        delete_empty_edges_sql = """
        DELETE FROM graph_edges
        WHERE cardinality(source_chunks) = 0
        RETURNING id
        """
        delete_empty_nodes_sql = """
        DELETE FROM graph_nodes
        WHERE cardinality(source_chunks) = 0
        RETURNING id
        """
        delete_orphan_edges_sql = """
        DELETE FROM graph_edges AS ge
        WHERE NOT EXISTS (SELECT 1 FROM graph_nodes AS gn WHERE gn.id = ge.src_id)
           OR NOT EXISTS (SELECT 1 FROM graph_nodes AS gn WHERE gn.id = ge.tgt_id)
        RETURNING id
        """

        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(prune_edges_sql)
            await conn.execute(prune_nodes_sql)
            empty_edge_rows = await conn.fetch(delete_empty_edges_sql)
            empty_node_rows = await conn.fetch(delete_empty_nodes_sql)
            orphan_edge_rows = await conn.fetch(delete_orphan_edges_sql)

        node_ids = [row["id"] for row in empty_node_rows]
        edge_ids = [row["id"] for row in [*empty_edge_rows, *orphan_edge_rows]]
        return list(dict.fromkeys(node_ids)), list(dict.fromkeys(edge_ids))

    # ---- chunks ----
    async def insert_chunks(self, chunks: list[Chunk]) -> None:
        sql = """
        INSERT INTO chunks(
            id, document_id, ord, heading_path, content, token_count,
            parent_id, parent_content, block_id
        )
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT(id) DO UPDATE SET
            content=EXCLUDED.content, heading_path=EXCLUDED.heading_path,
            token_count=EXCLUDED.token_count, parent_id=EXCLUDED.parent_id,
            parent_content=EXCLUDED.parent_content, block_id=EXCLUDED.block_id;
        """
        rows = [
            (
                c.id,
                c.document_id,
                c.ord,
                c.heading_path,
                c.content,
                c.token_count,
                c.parent_id,
                c.parent_content,
                c.block_id,
            )
            for c in chunks
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, rows)

    async def chunks_by_ids(self, chunk_ids: list[str]) -> list[Evidence]:
        """按 chunk id 批量取正文，转为 Evidence（图扩展用）。"""
        if not chunk_ids:
            return []
        sql = """
        SELECT c.id, c.parent_id, COALESCE(c.parent_content, c.content) AS content,
               c.heading_path, c.block_id, d.source, d.revision
        FROM chunks c JOIN documents d ON c.document_id = d.id
        WHERE c.id = ANY($1::text[])
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, chunk_ids)
        by_id = {r["id"]: r for r in rows}
        evidences: list[Evidence] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            evidence_key = row["parent_id"] or row["id"]
            evidence_id = make_evidence_id("chunk", evidence_key)
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            source = row["source"]
            if row["block_id"]:
                source = f"{source}#{row['block_id']}"
            evidences.append(
                Evidence(
                    evidence_id=evidence_id,
                    kind="chunk",
                    ref_id=row["id"],
                    text=row["content"],
                    source=source,
                    heading_path=row["heading_path"],
                    parent_id=row["parent_id"],
                    block_id=row["block_id"],
                    revision=row["revision"],
                )
            )
        return evidences

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
