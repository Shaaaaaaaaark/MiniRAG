"""Just-in-time retrieval from one explicitly referenced Feishu document."""
from __future__ import annotations

import logging
import math

from minirag.core.chunker import ParentChildChunker, parse_document
from minirag.models.factory import ModelBundle
from minirag.schemas import (
    Chunk,
    DocumentInput,
    Evidence,
    RetrievalResult,
    make_evidence_id,
)

from .feishu import FeishuCliClient

_logger = logging.getLogger(__name__)


class FeishuJitRetriever:
    def __init__(self, client: FeishuCliClient, models: ModelBundle) -> None:
        self._client = client
        self._models = models
        self._chunker = ParentChildChunker()

    async def retrieve(
        self,
        url_or_token: str,
        query: str,
        *,
        top_k: int = 8,
        enable_rerank: bool = True,
    ) -> RetrievalResult:
        document = await self._client.fetch_document(url_or_token)
        parsed = parse_document(
            DocumentInput(
                source=document.url,
                source_id=document.token,
                revision=document.revision_id,
                title=document.title,
                blocks=document.blocks,
            )
        )
        chunks = self._chunker.split(parsed)
        if not chunks:
            return RetrievalResult()

        if query.strip():
            query_vector = (await self._models.embedding.embed([query]))[0]
            child_vectors = await self._models.embedding.embed(
                [chunk.content for chunk in chunks]
            )
            scored = [
                (chunk, _cosine_similarity(query_vector, vector))
                for chunk, vector in zip(chunks, child_vectors)
            ]
            ranked = sorted(scored, key=lambda item: item[1], reverse=True)
            candidate_count = min(len(ranked), max(top_k * 4, top_k))
            candidates = ranked[:candidate_count]
        else:
            candidates = [(chunk, 0.0) for chunk in chunks]

        evidences = self._parent_evidences(
            document.url,
            document.revision_id,
            candidates,
        )
        if enable_rerank and query.strip() and evidences:
            try:
                rerank_results = await self._models.rerank.rerank(
                    query,
                    [evidence.text for evidence in evidences],
                    min(top_k, len(evidences)),
                )
                reranked: list[Evidence] = []
                for result in rerank_results:
                    if 0 <= result.index < len(evidences):
                        evidence = evidences[result.index]
                        evidence.score = result.score
                        reranked.append(evidence)
                if reranked:
                    evidences = reranked
            except Exception as error:  # noqa: BLE001 - rerank is optional
                _logger.warning("飞书 JIT rerank 失败，保留向量排序：%s", error)

        return RetrievalResult(chunks=evidences[:top_k])

    @staticmethod
    def _parent_evidences(
        source_url: str,
        revision: str,
        candidates: list[tuple[Chunk, float]],
    ) -> list[Evidence]:
        evidences: list[Evidence] = []
        seen: set[str] = set()
        for chunk, score in candidates:
            parent_key = chunk.parent_id or chunk.id
            evidence_id = make_evidence_id(parent_key)
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            source = source_url
            if chunk.block_id:
                source = f"{source_url}#{chunk.block_id}"
            evidences.append(
                Evidence(
                    evidence_id=evidence_id,
                    ref_id=chunk.id,
                    text=chunk.parent_content or chunk.content,
                    source=source,
                    heading_path=chunk.heading_path,
                    score=score,
                    parent_id=chunk.parent_id,
                    block_id=chunk.block_id,
                    revision=revision,
                )
            )
        return evidences


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
