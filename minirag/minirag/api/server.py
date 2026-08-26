"""FastAPI 接入层：检索即服务。

- POST /documents  提交文档索引（同步）
- POST /retrieve   检索证据（只检索、不生成）
- GET  /health     健康检查

启动时装配 MiniRAG 并连接存储，关闭时释放。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from minirag.rag import MiniRAG
from minirag.schemas import DocumentInput, Evidence, Keywords, QueryMode


class DocumentRequest(BaseModel):
    source: str
    text: str | None = None
    revision: str | None = None
    title: str | None = None
    gleaning: bool = False


class RetrieveRequest(BaseModel):
    query: str
    mode: QueryMode = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None
    enable_rerank: bool = True


class RetrieveResponse(BaseModel):
    keywords: Keywords = Field(default_factory=Keywords)
    entities: list[Evidence] = Field(default_factory=list)
    relationships: list[Evidence] = Field(default_factory=list)
    chunks: list[Evidence] = Field(default_factory=list)
    count: int = 0


_rag: MiniRAG | None = None


def get_rag() -> MiniRAG:
    if _rag is None:
        raise RuntimeError("MiniRAG 未初始化")
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    _rag = MiniRAG()
    await _rag.startup()
    try:
        yield
    finally:
        await _rag.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="MiniRAG Retrieval Service", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        rag = get_rag()
        return {
            "status": "ok",
            "chat_model": rag.settings.chat.model,
            "embedding_model": rag.settings.embedding.model,
            "rerank_model": rag.settings.rerank.model,
            "ready": True,
        }

    @app.post("/documents")
    async def index_document(req: DocumentRequest):
        rag = get_rag()
        doc = DocumentInput(source=req.source, text=req.text, revision=req.revision, title=req.title)
        return await rag.index(doc, gleaning=req.gleaning)

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        from minirag.schemas import QueryParam

        rag = get_rag()
        param = QueryParam(
            mode=req.mode,
            top_k=req.top_k,
            chunk_top_k=req.chunk_top_k,
            enable_rerank=req.enable_rerank,
        )
        result = await rag.retrieve(req.query, param)
        return RetrieveResponse(
            keywords=result.keywords,
            entities=result.entities,
            relationships=result.relationships,
            chunks=result.chunks,
            count=len(result.all_evidences),
        )

    return app


app = create_app()
