from pathlib import Path

from minirag_langchain.config import ChunkingConfig
from minirag_langchain.ingest import build_child_documents, parse_frontmatter


def test_parse_frontmatter() -> None:
    metadata, body = parse_frontmatter(
        '---\ntitle: "CloudWAN"\nrevision_id: "7"\n---\n# API\ncontent'
    )

    assert metadata == {"title": "CloudWAN", "revision_id": "7"}
    assert body.startswith("# API")


def test_build_child_documents_keeps_parent_and_source_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(
        """---
title: "CloudWAN API"
source_url: "https://example.test/docx/abc"
doc_token: "abc"
revision_id: "9"
---

<!-- feishu-block-id: heading-1 -->
# API

<!-- feishu-block-id: paragraph-1 -->
CreateGlobalNetwork accepts ClientToken. ClientToken is limited to 64 ASCII characters.
""",
        encoding="utf-8",
    )

    documents = build_child_documents(
        source,
        ChunkingConfig(
            parent_tokens=100,
            parent_overlap=10,
            child_tokens=30,
            child_overlap=5,
        ),
    )

    assert documents
    assert all(document.metadata["parent_id"] for document in documents)
    assert all(document.metadata["parent_content"] for document in documents)
    assert all(document.metadata["source"] == "https://example.test/docx/abc" for document in documents)
    assert all(document.metadata["revision"] == "9" for document in documents)
    assert all(document.metadata["heading_path"] == "API" for document in documents)
    assert "feishu-block-id" not in documents[0].page_content
