#!/usr/bin/env python3
"""minirag 最小验证脚本：索引语料 + 检索。

用法（在 minirag/ 目录，使 minirag 包可导入）：
  # 索引一篇语料
  python scripts/smoke.py index /path/to/document.md

  # 检索
  python scripts/smoke.py retrieve "BGP 中断如何处理" --top-k 8

需要：可用的 Milvus 与 PostgreSQL；模型 Key 经环境变量注入（见 config.yaml）。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 让 minirag 包可导入（脚本在 minirag/scripts/ 下）
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from minirag import DocumentInput, MiniRAG, QueryParam


async def cmd_index(args: argparse.Namespace) -> int:
    rag = MiniRAG()
    await rag.startup()
    try:
        report = await rag.index(DocumentInput(source=args.path))
        print(f"索引完成 doc={report.document_id} chunks={report.chunks}")
        return 0 if report.chunks > 0 else 1
    finally:
        await rag.shutdown()


async def cmd_retrieve(args: argparse.Namespace) -> int:
    rag = MiniRAG()
    await rag.startup()
    try:
        param = QueryParam(top_k=args.top_k)
        result = await rag.retrieve(args.query, param)
        for evidence in result.chunks:
            preview = " ".join(evidence.text.split())[:80]
            print(f"score={evidence.score:.4f} {evidence.evidence_id}  {preview}")
        return 0 if result.chunks else 1
    finally:
        await rag.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="minirag 冒烟验证")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="索引一篇 Markdown/TXT 语料")
    p_index.add_argument("path", help="语料文件路径")

    p_ret = sub.add_parser("retrieve", help="检索证据")
    p_ret.add_argument("query", help="查询问题")
    p_ret.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "index":
        return asyncio.run(cmd_index(args))
    return asyncio.run(cmd_retrieve(args))


if __name__ == "__main__":
    raise SystemExit(main())
