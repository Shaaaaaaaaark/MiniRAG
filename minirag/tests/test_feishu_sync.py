from pathlib import Path

import pytest

from minirag.integrations.feishu import FeishuDocument, FeishuDriveItem
from minirag.integrations.feishu_sync import FeishuFolderSync
from minirag.schemas import Block


class FakeFeishuClient:
    def __init__(self) -> None:
        self.fetch_count = 0

    async def list_folder(self, folder, *, recursive=True):
        return [
            FeishuDriveItem(
                token="doc-token",
                name="CloudWAN Design",
                type="docx",
                url="https://example.feishu.cn/docx/doc-token",
                modified_time="100",
                path="CloudWAN Design",
            )
        ]

    async def fetch_document(self, doc, *, title=None):
        self.fetch_count += 1
        return FeishuDocument(
            token="doc-token",
            url="https://example.feishu.cn/docx/doc-token",
            title=title or "CloudWAN Design",
            revision_id="7",
            blocks=[
                Block(type="heading", text="Design", level=1, block_id="h1"),
                Block(type="paragraph", text="Content", block_id="p1"),
            ],
            raw_xml="",
        )


@pytest.mark.asyncio
async def test_folder_sync_skips_unchanged_document(tmp_path: Path) -> None:
    client = FakeFeishuClient()
    synchronizer = FeishuFolderSync(client, tmp_path)

    first = await synchronizer.sync("folder-token")
    second = await synchronizer.sync("folder-token")

    assert first.fetched == 1
    assert second.skipped == 1
    assert client.fetch_count == 1
    assert (tmp_path / "docs" / "doc-token.md").exists()
    assert (tmp_path / "manifest.json").exists()
