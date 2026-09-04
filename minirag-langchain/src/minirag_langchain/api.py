"""FastAPI entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from minirag_langchain.schemas import RetrieveRequest, RetrieveResponse
from minirag_langchain.service import LangChainRAG

_rag: LangChainRAG | None = None


def get_rag() -> LangChainRAG:
    if _rag is None:
        raise RuntimeError("MiniRAG LangChain 未初始化")
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    _rag = LangChainRAG()
    yield


app = FastAPI(title="MiniRAG LangChain", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    rag = get_rag()
    return {
        "status": "ok",
        "ready": True,
        "framework": "langchain",
        "embedding_model": rag.settings.embedding.model,
        "rerank_model": rag.settings.rerank.model,
        "collection": rag.settings.milvus.collection,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    return await get_rag().retrieve(
        request.query,
        top_k=request.effective_top_k,
        enable_rerank=request.enable_rerank,
    )
