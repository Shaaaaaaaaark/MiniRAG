"""Retrieval orchestration: hybrid recall -> parent hydration -> rerank."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings

from minirag_langchain.config import Settings, load_settings
from minirag_langchain.ingest import build_corpus
from minirag_langchain.rerank import rerank
from minirag_langchain.schemas import Evidence, RetrieveResponse

_logger = logging.getLogger(__name__)


def hydrate_parents(
    child_hits: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Collapse child hits to their parent chunks, keeping the best-ranked hit."""
    parents: list[tuple[Document, float]] = []
    seen: set[str] = set()
    for child, score in child_hits:
        metadata = dict(child.metadata)
        parent_id = str(metadata.get("parent_id") or child.id or "")
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        parent_content = str(metadata.pop("parent_content", "") or child.page_content)
        metadata["child_id"] = str(metadata.get("child_id") or child.id or "")
        metadata["parent_id"] = parent_id
        parents.append(
            (Document(id=parent_id, page_content=parent_content, metadata=metadata), float(score))
        )
    return parents


def to_evidence(document: Document, score: float) -> Evidence:
    metadata = document.metadata
    parent_id = str(metadata["parent_id"])
    evidence_id = "e_" + hashlib.sha1(f"chunk:{parent_id}".encode()).hexdigest()[:16]
    source = str(metadata.get("source") or "")
    block_id = str(metadata.get("block_id") or "")
    if block_id and "#" not in source:
        source = f"{source}#{block_id}"
    return Evidence(
        evidence_id=evidence_id,
        ref_id=str(metadata.get("child_id") or ""),
        text=document.page_content,
        source=source,
        heading_path=str(metadata.get("heading_path") or ""),
        score=score,
        parent_id=parent_id,
        block_id=block_id,
        revision=str(metadata.get("revision") or ""),
    )


class LangChainRAG:
    """Wires LangChain embeddings + Milvus hybrid store into a retrieval pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.embeddings = self._build_embeddings()
        self.vectorstore = self._build_vectorstore(drop_old=False)

    def _build_embeddings(self) -> OpenAIEmbeddings:
        embedding = self.settings.embedding
        return OpenAIEmbeddings(
            model=embedding.model,
            dimensions=embedding.dimensions,
            api_key=embedding.api_key,
            base_url=embedding.base_url,
            timeout=embedding.timeout_seconds,
            max_retries=embedding.max_retries,
            chunk_size=embedding.batch_size,
            check_embedding_ctx_length=False,
        )

    def _build_vectorstore(self, *, drop_old: bool) -> Milvus:
        milvus = self.settings.milvus
        return Milvus(
            embedding_function=self.embeddings,
            builtin_function=BM25BuiltInFunction(
                analyzer_params={"type": "chinese"},
                function_name="langchain_bm25",
            ),
            collection_name=milvus.collection,
            collection_description="LangChain MiniRAG Parent-Child baseline",
            connection_args={"uri": milvus.uri, "db_name": milvus.db},
            consistency_level="Bounded",
            index_params=[
                {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
                {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
            ],
            search_params=[
                {"metric_type": "COSINE", "params": {"ef": 64}},
                {"metric_type": "BM25", "params": {}},
            ],
            vector_field=["dense", "sparse"],
            primary_field="pk",
            text_field="text",
            enable_dynamic_field=True,
            auto_id=False,
            drop_old=drop_old,
        )

    async def rebuild(self, source: Path) -> int:
        documents = build_corpus(source, self.settings.chunking)
        if not documents:
            raise ValueError(f"no Markdown documents found under {source}")
        ids = [str(document.id) for document in documents]
        self.vectorstore = self._build_vectorstore(drop_old=True)
        await asyncio.to_thread(self.vectorstore.add_documents, documents=documents, ids=ids)
        return len(documents)

    async def _recall(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Dense + BM25 hybrid recall fused by Milvus RRF, then parent hydration."""
        retrieval = self.settings.retrieval
        fetch_k = retrieval.candidate_top_k
        candidate_k = max(fetch_k, retrieval.rerank_top_k, top_k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            child_hits = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                query,
                k=candidate_k,
                fetch_k=fetch_k,
                ranker_type="rrf",
                ranker_params={"k": retrieval.rrf_k},
            )
        return hydrate_parents(child_hits)

    async def _rerank(
        self, query: str, ranked: list[tuple[Document, float]], top_k: int
    ) -> list[tuple[Document, float]]:
        """Optional precision stage; degrade to the RRF order on any failure."""
        try:
            reranked = await rerank(
                self.settings.rerank,
                query,
                [document for document, _ in ranked],
                max(top_k, self.settings.retrieval.rerank_top_k),
            )
            return reranked or ranked
        except Exception as error:  # noqa: BLE001 - rerank must degrade to RRF
            _logger.warning("rerank failed, preserving RRF order: %s", error)
            return ranked

    async def retrieve(
        self, query: str, *, top_k: int = 8, enable_rerank: bool = True
    ) -> RetrieveResponse:
        ranked = await self._recall(query, top_k)
        if enable_rerank and ranked:
            ranked = await self._rerank(query, ranked, top_k)
        chunks = [to_evidence(document, score) for document, score in ranked[:top_k]]
        return RetrieveResponse(chunks=chunks, count=len(chunks))
