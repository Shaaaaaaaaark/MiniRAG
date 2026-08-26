"""文档解析 + 分块。

解析：Markdown/TXT，按 ATX 标题(#)切 heading，其余为 paragraph。
分块：按标题层级分段，段内再按 token 递归切分，保留 heading_path。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from minirag.core.tokenizer import count_tokens, split_by_tokens
from minirag.schemas import Block, Chunk, DocumentInput, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _doc_id(source: str, revision: str | None) -> str:
    return "d_" + hashlib.sha1(f"{source}:{revision or ''}".encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_id: str, ord_: int) -> str:
    return "c_" + hashlib.sha1(f"{document_id}:{ord_}".encode("utf-8")).hexdigest()[:16]


def parse_document(doc_input: DocumentInput) -> ParsedDocument:
    """解析 Markdown/TXT 为 Block 序列。text 优先，否则按 source 路径读取。"""
    if doc_input.text is not None:
        text = doc_input.text
    else:
        text = Path(doc_input.source).read_text(encoding="utf-8")

    blocks: list[Block] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            joined = "\n".join(para).strip()
            if joined:
                blocks.append(Block(type="paragraph", text=joined))
            para.clear()

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush_para()
            blocks.append(Block(type="heading", text=m.group(2).strip(), level=len(m.group(1))))
        else:
            para.append(line)
    flush_para()

    title = doc_input.title or (Path(doc_input.source).stem if doc_input.source else None)
    return ParsedDocument(
        document_id=_doc_id(doc_input.source, doc_input.revision),
        source=doc_input.source,
        title=title,
        revision=doc_input.revision,
        blocks=blocks,
    )


class HeaderTokenChunker:
    """按标题层级分段，段内再按 token 递归切分，保留 heading_path。"""

    def __init__(self, max_tokens: int = 700, overlap: int = 100) -> None:
        self._max_tokens = max_tokens
        self._overlap = overlap

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        heading_stack: list[str] = []
        buffer: list[str] = []

        def heading_path() -> str:
            return "/".join(heading_stack)

        def flush() -> None:
            if not buffer:
                return
            text = "\n".join(buffer).strip()
            buffer.clear()
            if not text:
                return
            for piece in self._split_by_tokens(text):
                ord_ = len(chunks)
                chunks.append(
                    Chunk(
                        id=_chunk_id(doc.document_id, ord_),
                        document_id=doc.document_id,
                        ord=ord_,
                        heading_path=heading_path(),
                        content=piece,
                        token_count=count_tokens(piece),
                    )
                )

        for block in doc.blocks:
            if block.type == "heading":
                flush()
                self._push_heading(heading_stack, block)
            else:
                buffer.append(block.text)
        flush()
        return chunks

    @staticmethod
    def _push_heading(stack: list[str], block: Block) -> None:
        level = block.level or (len(stack) + 1)
        del stack[level - 1 :]
        stack.append(block.text.strip())

    def _split_by_tokens(self, text: str) -> list[str]:
        if count_tokens(text) <= self._max_tokens:
            return [text]

        paragraphs: list[str] = []
        for p in text.split("\n"):
            if not p.strip():
                continue
            if count_tokens(p) > self._max_tokens:
                paragraphs.extend(split_by_tokens(p, self._max_tokens))
            else:
                paragraphs.append(p)

        pieces: list[str] = []
        current: list[str] = []
        for para in paragraphs:
            trial = "\n".join(current + [para])
            if current and count_tokens(trial) > self._max_tokens:
                pieces.append("\n".join(current))
                current = self._carry_overlap(current) + [para]
            else:
                current.append(para)
        if current:
            pieces.append("\n".join(current))
        return pieces

    def _carry_overlap(self, current: list[str]) -> list[str]:
        carried: list[str] = []
        total = 0
        for para in reversed(current):
            t = count_tokens(para)
            if total + t > self._overlap:
                break
            carried.insert(0, para)
            total += t
        return carried
