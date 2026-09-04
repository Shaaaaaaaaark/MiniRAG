"""PostgreSQL 文档与 Parent-Child 分块存储。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from minirag.config import PostgresCfg
from minirag.schemas import Chunk, Evidence, ParsedDocument, make_evidence_id

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
            self._pool = None

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
        """按召回顺序回填 Parent 正文并去重。"""
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
            evidence_id = make_evidence_id(evidence_key)
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            source = row["source"]
            if row["block_id"]:
                source = f"{source}#{row['block_id']}"
            evidences.append(
                Evidence(
                    evidence_id=evidence_id,
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
