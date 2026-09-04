"""文档解析 + 分块。

解析：Markdown/TXT，按 ATX 标题(#)切 heading，其余为 paragraph。
分块：按标题层级分段，段内再按 token 递归切分，保留 heading_path。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from minirag.core.tokenizer import count_tokens, split_by_token_windows, split_by_tokens
from minirag.schemas import Block, Chunk, DocumentInput, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def document_id(source: str, source_id: str | None = None) -> str:
    """生成稳定文档 ID；revision 变化不得改变文档身份。"""
    key = source_id or source
    return "d_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_id: str, ord_: int) -> str:
    return "c_" + hashlib.sha1(f"{document_id}:{ord_}".encode()).hexdigest()[:16]


def _parent_id(document_id: str, ord_: int) -> str:
    return "p_" + hashlib.sha1(f"{document_id}:parent:{ord_}".encode()).hexdigest()[:16]


def _push_heading(stack: list[str], block: Block) -> None:
    level = block.level or (len(stack) + 1)
    del stack[level - 1 :]
    stack.append(block.text.strip())


def parse_document(doc_input: DocumentInput) -> ParsedDocument:
    """解析 Markdown/TXT 为 Block 序列。text 优先，否则按 source 路径读取。"""
    if doc_input.blocks is not None:
        blocks = [block.model_copy(deep=True) for block in doc_input.blocks]
    else:
        if doc_input.text is not None:
            text = doc_input.text
        else:
            text = Path(doc_input.source).read_text(encoding="utf-8")

        blocks = []
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
        document_id=document_id(doc_input.source, doc_input.source_id),
        source=doc_input.source,
        source_id=doc_input.source_id,
        title=title,
        revision=doc_input.revision,
        blocks=blocks,
    )


class ParentChildChunker:
    """技术文档分块：小块检索，命中后返回所属父章节。"""

    def __init__(
        self,
        parent_tokens: int = 900,
        child_tokens: int = 250,
        child_overlap: int = 40,
    ) -> None:
        if child_tokens >= parent_tokens:
            raise ValueError("child_tokens 必须小于 parent_tokens")
        self._parent_tokens = parent_tokens
        self._child_tokens = child_tokens
        self._child_overlap = child_overlap

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        heading_stack: list[str] = []
        section_blocks: list[Block] = []
        parent_ord = 0

        def flush_section() -> None:
            nonlocal parent_ord
            if not section_blocks:
                return
            path = "/".join(heading_stack)
            for body, block_id in self._split_parent_blocks(section_blocks):
                parent_id = _parent_id(doc.document_id, parent_ord)
                parent_ord += 1
                parent_content = self._with_context(doc.title, path, body)
                child_pieces = split_by_token_windows(
                    body,
                    self._child_tokens,
                    self._child_overlap,
                )
                for piece in child_pieces:
                    child_content = self._with_context(doc.title, path, piece)
                    ord_ = len(chunks)
                    chunks.append(
                        Chunk(
                            id=_chunk_id(doc.document_id, ord_),
                            document_id=doc.document_id,
                            ord=ord_,
                            heading_path=path,
                            content=child_content,
                            token_count=count_tokens(child_content),
                            parent_id=parent_id,
                            parent_content=parent_content,
                            block_id=block_id,
                        )
                    )
            section_blocks.clear()

        for block in doc.blocks:
            if block.type == "heading":
                flush_section()
                _push_heading(heading_stack, block)
            elif block.text.strip():
                section_blocks.append(block)
        flush_section()
        return chunks

    def _split_parent_blocks(self, blocks: list[Block]) -> list[tuple[str, str | None]]:
        parents: list[tuple[str, str | None]] = []
        current: list[str] = []
        current_block_id: str | None = None

        def flush() -> None:
            nonlocal current_block_id
            if current:
                parents.append(("\n\n".join(current).strip(), current_block_id))
            current.clear()
            current_block_id = None

        for block in blocks:
            pieces = (
                split_by_tokens(block.text, self._parent_tokens)
                if count_tokens(block.text) > self._parent_tokens
                else [block.text]
            )
            for piece in pieces:
                trial = "\n\n".join([*current, piece])
                if current and count_tokens(trial) > self._parent_tokens:
                    flush()
                if current_block_id is None:
                    current_block_id = block.block_id
                current.append(piece)
        flush()
        return parents

    @staticmethod
    def _with_context(title: str | None, heading_path: str, text: str) -> str:
        context = [part for part in (title, heading_path) if part]
        if not context:
            return text
        return "\n".join(context) + "\n\n" + text
