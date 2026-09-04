"""Command-line entry point."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from minirag_langchain.service import LangChainRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniRAG LangChain baseline")
    parser.add_argument("--config", type=Path, help="private YAML config; defaults to config.yaml")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index = subcommands.add_parser("index", help="rebuild the baseline collection")
    index.add_argument("source", type=Path)

    retrieve = subcommands.add_parser("retrieve", help="retrieve evidence")
    retrieve.add_argument("query")
    retrieve.add_argument("--top-k", type=int, default=8)
    retrieve.add_argument("--disable-rerank", action="store_true")

    serve = subcommands.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8091)
    return parser.parse_args()


async def run_retrieve(query: str, top_k: int, enable_rerank: bool) -> int:
    response = await LangChainRAG().retrieve(query, top_k=top_k, enable_rerank=enable_rerank)
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    return 0 if response.chunks else 1


def main() -> int:
    args = parse_args()
    if args.config:
        os.environ["MINIRAG_LANGCHAIN_CONFIG"] = str(args.config.resolve())

    if args.command == "index":
        rag = LangChainRAG()
        count = asyncio.run(rag.rebuild(args.source))
        print(f"indexed child_documents={count} collection={rag.settings.milvus.collection}")
        return 0
    if args.command == "retrieve":
        return asyncio.run(run_retrieve(args.query, args.top_k, not args.disable_rerank))

    import uvicorn

    uvicorn.run("minirag_langchain.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
