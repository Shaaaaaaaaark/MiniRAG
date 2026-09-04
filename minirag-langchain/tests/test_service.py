from langchain_core.documents import Document

from minirag_langchain.schemas import RetrieveRequest
from minirag_langchain.service import hydrate_parents


def test_request_prefers_chunk_top_k() -> None:
    request = RetrieveRequest(
        query="question",
        top_k=10,
        chunk_top_k=3,
    )

    assert request.effective_top_k == 3


def test_hydrate_parents_deduplicates_child_hits() -> None:
    first = Document(
        id="child-1",
        page_content="child one",
        metadata={
            "child_id": "child-1",
            "parent_id": "parent-1",
            "parent_content": "complete parent",
        },
    )
    second = Document(
        id="child-2",
        page_content="child two",
        metadata={
            "child_id": "child-2",
            "parent_id": "parent-1",
            "parent_content": "complete parent",
        },
    )

    parents = hydrate_parents([(first, 0.9), (second, 0.8)])

    assert len(parents) == 1
    assert parents[0][0].page_content == "complete parent"
    assert parents[0][0].metadata["child_id"] == "child-1"
    assert parents[0][1] == 0.9
