"""FastAPI 接入层：检索即服务。

- POST /documents  提交文档索引（同步）
- POST /retrieve   检索证据（只检索、不生成）
- GET  /health     健康检查

启动时装配 MiniRAG 并连接存储，关闭时释放。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from minirag.integrations.feishu import FeishuCliError, find_feishu_document_url
from minirag.rag import MiniRAG
from minirag.schemas import DocumentInput, Evidence, QueryParam, RetrievalResult


class DocumentRequest(BaseModel):
    source: str
    source_id: str | None = None
    text: str | None = None
    revision: str | None = None
    title: str | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    source_url: str | None = None
    mode: Literal["text"] = "text"
    top_k: int | None = Field(default=None, ge=1, le=50)
    chunk_top_k: int | None = Field(default=None, ge=1, le=50)
    enable_rerank: bool = True


class FeishuRetrieveRequest(BaseModel):
    source_url: str
    query: str = ""
    top_k: int = Field(default=8, ge=1, le=50)
    enable_rerank: bool = True


class RetrieveResponse(BaseModel):
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
            "embedding_model": rag.settings.embedding.model,
            "rerank_model": rag.settings.rerank.model,
            "ready": True,
        }

    @app.post("/documents")
    async def index_document(req: DocumentRequest):
        rag = get_rag()
        doc = DocumentInput(
            source=req.source,
            source_id=req.source_id,
            text=req.text,
            revision=req.revision,
            title=req.title,
        )
        return await rag.index(doc)

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        rag = get_rag()
        source_url = req.source_url or find_feishu_document_url(req.query)
        if source_url:
            jit_query = req.query.replace(source_url, " ").strip()
            top_k = req.chunk_top_k or req.top_k or rag.settings.retrieval.chunk_top_k
            try:
                result = await rag.retrieve_feishu(
                    source_url,
                    jit_query,
                    top_k=top_k,
                    enable_rerank=req.enable_rerank,
                )
            except (FeishuCliError, RuntimeError, ValueError) as err:
                raise HTTPException(status_code=503, detail=str(err)) from err
            return _retrieval_response(result)

        param = QueryParam(
            top_k=req.chunk_top_k or req.top_k,
            enable_rerank=req.enable_rerank,
        )
        result = await rag.retrieve(req.query, param)
        return _retrieval_response(result)

    @app.post("/feishu/retrieve", response_model=RetrieveResponse)
    async def retrieve_feishu(req: FeishuRetrieveRequest) -> RetrieveResponse:
        rag = get_rag()
        try:
            result = await rag.retrieve_feishu(
                req.source_url,
                req.query,
                top_k=req.top_k,
                enable_rerank=req.enable_rerank,
            )
        except (FeishuCliError, RuntimeError, ValueError) as err:
            raise HTTPException(status_code=503, detail=str(err)) from err
        return _retrieval_response(result)

    return app


def _retrieval_response(result: RetrievalResult) -> RetrieveResponse:
    return RetrieveResponse(
        chunks=result.chunks,
        count=len(result.chunks),
    )


app = create_app()
