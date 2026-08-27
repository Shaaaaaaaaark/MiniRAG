"""Feishu Drive/Docx integration backed by the authenticated ``lark-cli``.

The integration keeps transport and document normalization outside the RAG core:

- Drive folders are listed recursively with explicit pagination.
- Docx content is fetched as XML with block IDs.
- XML blocks are normalized into MiniRAG ``Block`` objects without regex-based
  blanket tag removal.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minirag.schemas import Block

_DOC_URL_RE = re.compile(
    r"https?://[^\s`]+/(?:docx|wiki)/[A-Za-z0-9_-]+(?:[?#][^\s`]*)?"
)
_DOC_TOKEN_RE = re.compile(r"/(?:docx|wiki)/([A-Za-z0-9_-]+)")
_FOLDER_TOKEN_RE = re.compile(r"/drive/folder/([A-Za-z0-9_-]+)")
_HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}
_MEDIA_TAGS = {
    "img": "图片",
    "whiteboard": "画板",
    "sheet": "电子表格",
    "bitable": "多维表格",
    "source": "附件",
    "task": "任务",
    "base_refer": "引用",
    "base-refer": "引用",
    "synced_reference": "同步块",
    "synced-reference": "同步块",
}


class FeishuCliError(RuntimeError):
    """Raised when lark-cli returns a failed envelope or invalid JSON."""


@dataclass(frozen=True)
class FeishuDriveItem:
    token: str
    name: str
    type: str
    url: str
    parent_token: str = ""
    owner_id: str = ""
    created_time: str = ""
    modified_time: str = ""
    path: str = ""


@dataclass(frozen=True)
class FeishuDocument:
    token: str
    url: str
    title: str
    revision_id: str
    blocks: list[Block]
    raw_xml: str


def find_feishu_document_url(text: str) -> str | None:
    match = _DOC_URL_RE.search(text)
    return match.group(0).rstrip(".,;，。；)") if match else None


def document_token(value: str) -> str:
    match = _DOC_TOKEN_RE.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    raise ValueError(f"无法识别飞书文档 token: {value}")


def folder_token(value: str) -> str:
    match = _FOLDER_TOKEN_RE.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    raise ValueError(f"无法识别飞书文件夹 token: {value}")


class FeishuCliClient:
    def __init__(
        self,
        cli_path: str = "lark-cli",
        identity: str = "user",
        timeout_seconds: float = 120.0,
    ) -> None:
        if identity not in {"user", "bot"}:
            raise ValueError("Feishu identity 只能是 user 或 bot")
        self._cli_path = cli_path
        self._identity = identity
        self._timeout_seconds = timeout_seconds

    async def list_folder(
        self,
        folder: str,
        *,
        recursive: bool = True,
    ) -> list[FeishuDriveItem]:
        root_token = folder_token(folder)
        queue: list[tuple[str, str]] = [(root_token, "")]
        items: list[FeishuDriveItem] = []
        seen: set[tuple[str, str]] = set()

        while queue:
            current_token, current_path = queue.pop(0)
            page_token: str | None = None
            while True:
                params: dict[str, Any] = {
                    "folder_token": current_token,
                    "page_size": 200,
                }
                if page_token:
                    params["page_token"] = page_token
                payload = await self._run(
                    "drive",
                    "files",
                    "list",
                    "--as",
                    self._identity,
                    "--params",
                    json.dumps(params, ensure_ascii=False),
                    "--format",
                    "json",
                )
                data = payload.get("data") or {}
                for raw in data.get("files") or []:
                    token = str(raw.get("token") or "")
                    kind = str(raw.get("type") or "")
                    if not token or not kind:
                        continue
                    dedupe_key = (kind, token)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    name = str(raw.get("name") or token)
                    path = f"{current_path}/{name}".strip("/")
                    item = FeishuDriveItem(
                        token=token,
                        name=name,
                        type=kind,
                        url=str(raw.get("url") or ""),
                        parent_token=str(raw.get("parent_token") or current_token),
                        owner_id=str(raw.get("owner_id") or ""),
                        created_time=str(raw.get("created_time") or ""),
                        modified_time=str(raw.get("modified_time") or ""),
                        path=path,
                    )
                    items.append(item)
                    if recursive and kind == "folder":
                        queue.append((token, path))

                if data.get("has_more") is not True:
                    break
                next_page = data.get("next_page_token")
                if not next_page:
                    raise FeishuCliError(
                        f"飞书目录分页缺少 next_page_token: folder={current_token}"
                    )
                page_token = str(next_page)
        return items

    async def fetch_document(
        self,
        doc: str,
        *,
        title: str | None = None,
    ) -> FeishuDocument:
        payload = await self._run(
            "docs",
            "+fetch",
            "--as",
            self._identity,
            "--doc",
            doc,
            "--detail",
            "with-ids",
            "--doc-format",
            "xml",
            "--format",
            "json",
        )
        document = (payload.get("data") or {}).get("document") or {}
        raw_xml = str(document.get("content") or "")
        if not raw_xml:
            raise FeishuCliError(f"飞书文档正文为空: {doc}")
        parsed_title, blocks = parse_feishu_xml(raw_xml)
        token = str(document.get("document_id") or document_token(doc))
        revision_id = str(document.get("revision_id") or "")
        canonical_url = (
            doc.split("#", 1)[0].split("?", 1)[0]
            if doc.startswith("http")
            else f"https://www.feishu.cn/docx/{token}"
        )
        return FeishuDocument(
            token=token,
            url=canonical_url,
            title=title or parsed_title or token,
            revision_id=revision_id,
            blocks=blocks,
            raw_xml=raw_xml,
        )

    async def _run(self, *args: str) -> dict[str, Any]:
        env = os.environ.copy()
        env["LARKSUITE_CLI_NO_UPDATE_NOTIFIER"] = "1"
        env["LARKSUITE_CLI_NO_SKILLS_NOTIFIER"] = "1"
        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as err:
            raise FeishuCliError(
                f"找不到 {self._cli_path}，请先安装并登录 lark-cli"
            ) from err

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as err:
            process.kill()
            await process.communicate()
            raise FeishuCliError(
                f"lark-cli 执行超时（{self._timeout_seconds}s）"
            ) from err

        output = stdout.decode("utf-8", errors="replace").strip()
        error_output = stderr.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(output or error_output)
        except json.JSONDecodeError as err:
            detail = error_output or output or f"exit={process.returncode}"
            raise FeishuCliError(f"lark-cli 返回非 JSON: {detail[:500]}") from err
        if process.returncode != 0 or payload.get("ok") is not True:
            error = payload.get("error") or {}
            message = error.get("message") or error_output or "unknown lark-cli error"
            raise FeishuCliError(str(message))
        return payload


def parse_feishu_xml(content: str) -> tuple[str, list[Block]]:
    """Convert Docx XML into canonical blocks while preserving block IDs."""
    xml = content.strip()
    if xml.startswith("<?xml"):
        xml = xml.split("?>", 1)[1]
    try:
        root = ET.fromstring(f"<root>{xml}</root>")
    except ET.ParseError as err:
        raise ValueError(f"飞书文档 XML 无法解析: {err}") from err

    title = ""
    blocks: list[Block] = []

    def visit(node: ET.Element) -> None:
        nonlocal title
        tag = _tag(node)
        if tag == "title":
            title = _clean_text(_inline_text(node))
            return
        if tag in _HEADING_TAGS:
            text = _clean_text(_inline_text(node))
            if text:
                blocks.append(
                    Block(
                        type="heading",
                        text=text,
                        level=_HEADING_TAGS[tag],
                        block_id=node.get("id"),
                    )
                )
            return
        if tag == "table":
            text = _render_table(node)
            if text:
                blocks.append(
                    Block(type="table", text=text, block_id=node.get("id"))
                )
            return
        if tag == "pre":
            code = _clean_text(_inline_text(node))
            if code:
                lang = node.get("lang") or ""
                blocks.append(
                    Block(
                        type="paragraph",
                        text=f"```{lang}\n{code}\n```",
                        block_id=node.get("id"),
                    )
                )
            return
        if tag in {"ul", "ol"}:
            text = _render_list(node, ordered=tag == "ol")
            if text:
                blocks.append(
                    Block(type="paragraph", text=text, block_id=node.get("id"))
                )
            return
        if tag in {"p", "blockquote"}:
            text = _clean_text(_inline_text(node))
            if text:
                prefix = "> " if tag == "blockquote" else ""
                blocks.append(
                    Block(
                        type="paragraph",
                        text=prefix + text,
                        block_id=node.get("id"),
                    )
                )
            return
        if tag in _MEDIA_TAGS:
            text = _media_placeholder(node)
            blocks.append(
                Block(type="paragraph", text=text, block_id=node.get("id"))
            )
            return
        for child in node:
            visit(child)

    for child in root:
        visit(child)
    return title, _dedupe_empty_blocks(blocks)


def blocks_to_markdown(document: FeishuDocument) -> str:
    metadata = {
        "source_type": "feishu_docx",
        "title": document.title,
        "source_url": document.url,
        "doc_token": document.token,
        "revision_id": document.revision_id,
    }
    lines = ["---"]
    for key, value in metadata.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.extend(["---", ""])
    for block in document.blocks:
        if block.block_id:
            lines.append(f"<!-- feishu-block-id: {block.block_id} -->")
        if block.type == "heading":
            level = min(max(block.level or 1, 1), 6)
            lines.append(f"{'#' * level} {block.text}")
        else:
            lines.append(block.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _tag(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1].lower()


def _inline_text(node: ET.Element) -> str:
    parts: list[str] = [node.text or ""]
    for child in node:
        tag = _tag(child)
        if tag == "br":
            parts.append("\n")
        elif tag in _MEDIA_TAGS:
            parts.append(_media_placeholder(child))
        elif tag == "cite":
            inner = _inline_text(child).strip()
            parts.append(
                inner
                or child.get("title")
                or child.get("user-name")
                or child.get("url")
                or ""
            )
        elif tag == "checkbox":
            done = str(child.get("done") or "").lower() == "true"
            parts.append(("[x] " if done else "[ ] ") + _inline_text(child))
        else:
            parts.append(_inline_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _render_table(table: ET.Element) -> str:
    rows: list[list[str]] = []
    header_index: int | None = None
    for row in table.iter():
        if _tag(row) != "tr":
            continue
        cells: list[str] = []
        has_header = False
        for cell in row:
            if _tag(cell) not in {"td", "th"}:
                continue
            has_header = has_header or _tag(cell) == "th"
            cells.append(_escape_table_cell(_clean_text(_inline_text(cell))))
        if cells:
            if has_header and header_index is None:
                header_index = len(rows)
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header_index = header_index if header_index is not None else 0
    header = normalized[header_index]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for index, row in enumerate(normalized)
        if index != header_index
    )
    return "\n".join(lines)


def _render_list(node: ET.Element, *, ordered: bool) -> str:
    lines: list[str] = []
    index = 1
    for child in node:
        if _tag(child) != "li":
            continue
        text = _clean_text(_inline_text(child))
        if text:
            prefix = f"{index}. " if ordered else "- "
            lines.append(prefix + text)
            index += 1
    return "\n".join(lines)


def _media_placeholder(node: ET.Element) -> str:
    tag = _tag(node)
    label = _MEDIA_TAGS.get(tag, tag)
    name = (
        node.get("title")
        or node.get("name")
        or node.get("token")
        or node.get("src-token")
        or node.get("url")
        or ""
    )
    return f"[{label}: {name}]" if name else f"[{label}]"


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _dedupe_empty_blocks(blocks: list[Block]) -> list[Block]:
    output: list[Block] = []
    for block in blocks:
        if not block.text.strip():
            continue
        if (
            output
            and block.type != "heading"
            and output[-1].type == block.type
            and output[-1].text == block.text
        ):
            continue
        output.append(block)
    return output


def write_document_markdown(document: FeishuDocument, path: Path) -> str:
    """Atomically write normalized Markdown and return its content."""
    markdown = blocks_to_markdown(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(path)
    return markdown
