#!/usr/bin/env python3
"""minirag 最小验证脚本：索引语料 + 检索。

用法（在 minirag/ 目录，使 minirag 包可导入）：
  # 索引一篇语料
  python scripts/smoke.py index ../corpus/云网络告警处理手册.md

  # 检索（默认 mix 模式）
  python scripts/smoke.py retrieve "BGP 中断如何处理" --mode mix --top-k 8

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
        print(
            f"索引完成 doc={report.document_id} "
            f"chunks={report.chunks} entities={report.entities} relations={report.relations}"
        )
        return 0 if report.chunks > 0 else 1
    finally:
        await rag.shutdown()


async def cmd_retrieve(args: argparse.Namespace) -> int:
    rag = MiniRAG()
    await rag.startup()
    try:
        param = QueryParam(mode=args.mode, top_k=args.top_k)
        result = await rag.retrieve(args.query, param)
        kw = result.keywords
        print(f"模式={args.mode}  关键词 high={kw.high_level} low={kw.low_level}")
        print(f"实体 {len(result.entities)} | 关系 {len(result.relationships)} | chunk {len(result.chunks)}\n")
        for group_name, evs in (
            ("实体", result.entities),
            ("关系", result.relationships),
            ("chunk", result.chunks),
        ):
            for ev in evs:
                preview = " ".join(ev.text.split())[:80]
                print(f"[{group_name}] score={ev.score:.4f} {ev.evidence_id}  {preview}")
        return 0 if result.all_evidences else 1
    finally:
        await rag.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="minirag 冒烟验证")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="索引一篇 Markdown/TXT 语料")
    p_index.add_argument("path", help="语料文件路径")

    p_ret = sub.add_parser("retrieve", help="检索证据")
    p_ret.add_argument("query", help="查询问题")
    p_ret.add_argument("--mode", default="mix", choices=["local", "global", "hybrid", "mix", "naive"])
    p_ret.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "index":
        return asyncio.run(cmd_index(args))
    return asyncio.run(cmd_retrieve(args))


if __name__ == "__main__":
    raise SystemExit(main())
