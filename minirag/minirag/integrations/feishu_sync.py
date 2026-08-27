"""Incremental Feishu folder synchronization for MiniRAG."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from minirag.integrations.feishu import (
    FeishuCliClient,
    FeishuDriveItem,
    folder_token,
    write_document_markdown,
)
from minirag.schemas import DocumentInput

if TYPE_CHECKING:
    from minirag.rag import MiniRAG


@dataclass
class FeishuSyncReport:
    discovered: int = 0
    fetched: int = 0
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    failed: int = 0


class FeishuFolderSync:
    def __init__(
        self,
        client: FeishuCliClient,
        output_dir: str | Path,
    ) -> None:
        self._client = client
        self._output_dir = Path(output_dir)
        self._docs_dir = self._output_dir / "docs"
        self._manifest_path = self._output_dir / "manifest.json"

    async def sync(
        self,
        folder: str,
        *,
        rag: MiniRAG | None = None,
        recursive: bool = True,
        prune: bool = False,
        gleaning: bool = False,
    ) -> FeishuSyncReport:
        report = FeishuSyncReport()
        manifest = self._load_manifest()
        previous_docs: dict[str, dict[str, Any]] = manifest.get("documents") or {}
        current_docs: dict[str, dict[str, Any]] = dict(previous_docs)

        items = await self._client.list_folder(folder, recursive=recursive)
        documents = [item for item in items if item.type == "docx"]
        report.discovered = len(documents)
        seen_tokens = {item.token for item in documents}

        for item in documents:
            previous = previous_docs.get(item.token) or {}
            output_path = self._docs_dir / f"{item.token}.md"
            is_unchanged = (
                previous.get("modified_time") == item.modified_time
                and previous.get("title") == item.name
                and previous.get("path") == item.path
                and previous.get("url") == item.url
                and output_path.exists()
                and not previous.get("last_error")
                and not previous.get("removed")
            )
            index_is_current = (
                rag is None
                or (
                    previous.get("revision_id")
                    and previous.get("indexed_revision")
                    == previous.get("revision_id")
                )
            )
            if is_unchanged and index_is_current:
                report.skipped += 1
                continue

            try:
                document = await self._client.fetch_document(
                    item.url or item.token,
                    title=item.name,
                )
                markdown = write_document_markdown(document, output_path)
                report.fetched += 1

                indexed_revision = previous.get("indexed_revision")
                if rag is not None:
                    await rag.index(
                        DocumentInput(
                            source=document.url,
                            source_id=document.token,
                            revision=document.revision_id,
                            title=document.title,
                            blocks=document.blocks,
                            chunking_strategy="parent_child",
                        ),
                        gleaning=gleaning,
                    )
                    indexed_revision = document.revision_id
                    report.indexed += 1

                current_docs[item.token] = {
                    **_item_metadata(item),
                    "revision_id": document.revision_id,
                    "content_sha256": hashlib.sha256(
                        markdown.encode("utf-8")
                    ).hexdigest(),
                    "local_path": str(output_path.relative_to(self._output_dir)),
                    "indexed_revision": indexed_revision,
                    "removed": False,
                }
                self._write_manifest(
                    folder,
                    current_docs,
                )
            except Exception as err:  # noqa: BLE001 - isolate failures per document
                report.failed += 1
                current_docs[item.token] = {
                    **previous,
                    **_item_metadata(item),
                    "last_error": str(err),
                    "removed": False,
                }
                self._write_manifest(folder, current_docs)

        removed_tokens = set(previous_docs) - seen_tokens
        for token in sorted(removed_tokens):
            entry = dict(current_docs[token])
            entry["removed"] = True
            current_docs[token] = entry
            if not prune:
                continue
            local_path = self._output_dir / str(entry.get("local_path") or "")
            if local_path.is_file():
                local_path.unlink()
            if rag is not None:
                source = str(entry.get("url") or token)
                await rag.delete_document(source=source, source_id=token)
            current_docs.pop(token, None)
            report.removed += 1

        self._write_manifest(folder, current_docs)
        return report

    def _load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return {"schema_version": 1, "documents": {}}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise ValueError(f"无法读取飞书同步 manifest: {self._manifest_path}") from err

    def _write_manifest(
        self,
        folder: str,
        documents: dict[str, dict[str, Any]],
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "folder_token": folder_token(folder),
            "folder": folder,
            "documents": documents,
        }
        temporary = self._manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self._manifest_path)


def _item_metadata(item: FeishuDriveItem) -> dict[str, Any]:
    data = asdict(item)
    return {
        "token": data["token"],
        "title": data["name"],
        "type": data["type"],
        "url": data["url"],
        "path": data["path"],
        "owner_id": data["owner_id"],
        "created_time": data["created_time"],
        "modified_time": data["modified_time"],
    }
