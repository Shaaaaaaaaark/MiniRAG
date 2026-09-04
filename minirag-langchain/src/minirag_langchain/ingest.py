"""Build Parent-Child LangChain Documents from normalized Feishu Markdown."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from minirag_langchain.config import ChunkingConfig

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_BLOCK_MARKER = re.compile(r"<!-- feishu-block-id: ([^ ]+) -->")
_HEADER_LEVELS = [(f"{'#' * level}", f"h{level}") for level in range(1, 7)]


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if match is None:
        return {}, text
    raw = yaml.safe_load(match.group(1)) or {}
    if not isinstance(raw, dict):
        raise TypeError("Markdown frontmatter 必须是键值对象")
    metadata = {str(key): str(value) for key, value in raw.items() if value is not None}
    return metadata, text[match.end() :]


def clean_markers(text: str) -> str:
    return _BLOCK_MARKER.sub("", text).strip()


def first_block_id(text: str) -> str:
    match = _BLOCK_MARKER.search(text)
    return match.group(1) if match else ""


def _heading_path(section_metadata: dict[str, str]) -> str:
    return "/".join(
        str(section_metadata.get(name, ""))
        for _, name in _HEADER_LEVELS
        if section_metadata.get(name)
    )


def build_child_documents(path: Path, config: ChunkingConfig) -> list[Document]:
    frontmatter, markdown = parse_frontmatter(path.read_text(encoding="utf-8"))
    source = frontmatter.get("source_url") or str(path.resolve())
    doc_token = frontmatter.get("doc_token") or stable_id("doc", source)
    revision = frontmatter.get("revision_id", "")
    title = frontmatter.get("title") or path.stem
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADER_LEVELS, return_each_line=False, strip_headers=False
    )
    parent_splitter = _token_splitter(config.parent_tokens, config.parent_overlap)
    child_splitter = _token_splitter(config.child_tokens, config.child_overlap)

    children: list[Document] = []
    parents = _parent_parts(markdown, header_splitter, parent_splitter)
    for parent_number, (heading_path, raw_parent) in enumerate(parents):
        parent_content = clean_markers(raw_parent)
        parent_id = stable_id(
            "parent", doc_token, revision, str(parent_number), parent_content
        )
        parent_block_id = first_block_id(raw_parent)
        for child_number, raw_child in enumerate(child_splitter.split_text(raw_parent)):
            child_content = clean_markers(raw_child)
            if not child_content:
                continue
            child_id = stable_id("child", parent_id, str(child_number), child_content)
            children.append(
                Document(
                    id=child_id,
                    page_content=child_content,
                    metadata={
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "source": source,
                        "heading_path": heading_path,
                        "block_id": first_block_id(raw_child) or parent_block_id,
                        "revision": revision,
                        "doc_token": doc_token,
                        "title": title,
                    },
                )
            )
    return children


def _parent_parts(
    markdown: str,
    header_splitter: MarkdownHeaderTextSplitter,
    parent_splitter: RecursiveCharacterTextSplitter,
) -> Iterator[tuple[str, str]]:
    for section in header_splitter.split_text(markdown):
        heading_path = _heading_path(section.metadata)
        for raw_parent in parent_splitter.split_text(section.page_content):
            if clean_markers(raw_parent):
                yield heading_path, raw_parent


def _token_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )


def build_corpus(source: Path, config: ChunkingConfig) -> list[Document]:
    paths = [source] if source.is_file() else sorted(source.rglob("*.md"))
    return [
        document
        for path in paths
        for document in build_child_documents(path, config)
    ]
