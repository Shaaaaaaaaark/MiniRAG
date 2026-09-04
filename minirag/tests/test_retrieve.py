import pytest

from minirag.config import RetrievalCfg
from minirag.core.retrieve import Retriever
from minirag.schemas import Evidence, QueryParam


class FakeEmbedding:
    dimensions = 2

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeRerank:
    async def rerank(self, query, documents, top_k):
        return []


class FakeModels:
    embedding = FakeEmbedding()
    rerank = FakeRerank()


class FakeMilvus:
    async def hybrid_search_chunks(self, query_text, query_vec, dense_k, bm25_k, rrf_k):
        return []


class FakePg:
    async def chunks_by_ids(self, chunk_ids):
        return []


@pytest.mark.asyncio
async def test_empty_recall_returns_empty_result() -> None:
    retriever = Retriever(
        RetrievalCfg(),
        FakeModels(),
        FakePg(),
        FakeMilvus(),
    )

    result = await retriever.retrieve(
        "BGP session down",
        QueryParam(enable_rerank=False),
    )

    assert result.chunks == []


class FailRerank:
    async def rerank(self, query, documents, top_k):
        raise TimeoutError("rerank unavailable")


class ModelsWithFailingRerank(FakeModels):
    rerank = FailRerank()


class MilvusWithHit:
    async def hybrid_search_chunks(self, query_text, query_vec, dense_k, bm25_k, rrf_k):
        return [
            Evidence(
                evidence_id="child-evidence",
                kind="chunk",
                ref_id="child-1",
                text="child",
                source="milvus",
                score=0.8,
            )
        ]


class PgWithParent:
    async def chunks_by_ids(self, chunk_ids):
        return [
            Evidence(
                evidence_id="parent-evidence",
                kind="chunk",
                ref_id="child-1",
                text="complete parent context",
                source="doc#block",
                parent_id="parent-1",
            )
        ]


@pytest.mark.asyncio
async def test_text_mode_falls_back_when_rerank_fails() -> None:
    retriever = Retriever(
        RetrievalCfg(),
        ModelsWithFailingRerank(),
        PgWithParent(),
        MilvusWithHit(),
    )

    result = await retriever.retrieve("BGP session down")

    assert [chunk.text for chunk in result.chunks] == ["complete parent context"]
