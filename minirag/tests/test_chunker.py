from minirag.core.chunker import ParentChildChunker, document_id, parse_document
from minirag.core.index import _graph_extraction_chunks
from minirag.core.tokenizer import split_by_token_windows
from minirag.schemas import Block, DocumentInput


def test_document_id_does_not_depend_on_revision() -> None:
    first = parse_document(
        DocumentInput(
            source="https://example.feishu.cn/docx/abc",
            source_id="abc",
            revision="1",
            text="# Title\nfirst",
        )
    )
    second = parse_document(
        DocumentInput(
            source="https://example.feishu.cn/docx/abc",
            source_id="abc",
            revision="2",
            text="# Title\nsecond",
        )
    )

    assert first.document_id == second.document_id == document_id(first.source, "abc")


def test_parent_child_chunker_keeps_parent_and_block_id() -> None:
    document = parse_document(
        DocumentInput(
            source="https://example.feishu.cn/docx/abc",
            source_id="abc",
            title="CloudWAN Design",
            blocks=[
                Block(type="heading", text="API", level=1, block_id="heading-1"),
                Block(
                    type="paragraph",
                    text="CreateCloudWAN accepts a project identifier. " * 80,
                    block_id="paragraph-1",
                ),
            ],
        )
    )

    chunks = ParentChildChunker(
        parent_tokens=200,
        child_tokens=60,
        child_overlap=10,
    ).split(document)

    assert len(chunks) > 1
    assert all(chunk.parent_id for chunk in chunks)
    assert all(chunk.parent_content for chunk in chunks)
    assert all(chunk.heading_path == "API" for chunk in chunks)
    assert chunks[0].block_id == "paragraph-1"
    assert chunks[0].parent_content.startswith("CloudWAN Design\nAPI\n\n")


def test_token_windows_do_not_emit_redundant_tail() -> None:
    text = "token " * 55

    windows = split_by_token_windows(text, size=60, overlap=10)

    assert len(windows) == 1


def test_graph_extraction_uses_each_parent_once() -> None:
    document = parse_document(
        DocumentInput(
            source="doc",
            title="Doc",
            blocks=[
                Block(type="heading", text="Section", level=1),
                Block(type="paragraph", text="content " * 300, block_id="p1"),
            ],
        )
    )
    children = ParentChildChunker(
        parent_tokens=180,
        child_tokens=50,
        child_overlap=10,
    ).split(document)

    extraction_chunks = _graph_extraction_chunks(children)

    assert len(children) > len(extraction_chunks)
    assert len(extraction_chunks) == len({chunk.parent_id for chunk in children})
    assert all(chunk.content == chunk.parent_content for chunk in extraction_chunks)
