#!/usr/bin/env python3
"""Synchronize a Feishu Drive folder into normalized Markdown and MiniRAG.

Examples:
  python scripts/sync_feishu.py --folder "https://.../drive/folder/..."
  python scripts/sync_feishu.py --folder "..." --index
  python scripts/sync_feishu.py --folder "..." --index --prune
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from minirag.integrations.feishu import FeishuCliClient
from minirag.integrations.feishu_sync import FeishuFolderSync
from minirag.rag import MiniRAG


async def run(args: argparse.Namespace) -> int:
    client = FeishuCliClient(
        cli_path=args.lark_cli,
        identity=args.identity,
        timeout_seconds=args.timeout,
    )
    synchronizer = FeishuFolderSync(client, args.output_dir)
    rag: MiniRAG | None = MiniRAG() if args.index else None
    if rag is not None:
        await rag.startup()
    try:
        report = await synchronizer.sync(
            args.folder,
            rag=rag,
            recursive=not args.no_recursive,
            prune=args.prune,
        )
    finally:
        if rag is not None:
            await rag.shutdown()

    print(
        "Feishu sync complete: "
        f"discovered={report.discovered} fetched={report.fetched} "
        f"indexed={report.indexed} skipped={report.skipped} "
        f"removed={report.removed} failed={report.failed}"
    )
    print(f"Output: {Path(args.output_dir).resolve()}")
    return 1 if report.failed else 0


def main() -> int:
    default_output = Path(__file__).resolve().parents[2] / "corpus" / "feishu"
    parser = argparse.ArgumentParser(
        description="Synchronize Feishu Docx files into MiniRAG corpus"
    )
    parser.add_argument("--folder", required=True, help="Feishu folder URL or token")
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        help="Normalized corpus and manifest directory",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index changed documents into MiniRAG after fetching",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove documents no longer present in the folder",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only synchronize direct children of the folder",
    )
    parser.add_argument("--identity", choices=["user", "bot"], default="user")
    parser.add_argument("--lark-cli", default="lark-cli")
    parser.add_argument("--timeout", type=float, default=120.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
