#!/usr/bin/env python3
"""Run retrieval-only Ragas evaluation against the MiniRAG HTTP API."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _ROOT / "minirag"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from minirag.config import load_settings


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    answerable: bool
    category: str
    difficulty: str
    gold_evidence: list[dict[str, Any]]

    @property
    def reference(self) -> str:
        return "\n\n".join(
            str(item.get("gold_text") or "").strip()
            for item in self.gold_evidence
            if str(item.get("gold_text") or "").strip()
        )

    @property
    def gold_block_ids(self) -> set[str]:
        return {
            str(item["block_id"])
            for item in self.gold_evidence
            if item.get("block_id")
        }


def load_cases(
    path: Path,
    *,
    categories: set[str],
    limit: int,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        case = BenchmarkCase(
            id=str(raw.get("id") or ""),
            question=str(raw.get("question") or ""),
            answerable=bool(raw.get("answerable")),
            category=str(raw.get("category") or ""),
            difficulty=str(raw.get("difficulty") or ""),
            gold_evidence=list(raw.get("gold_evidence") or []),
        )
        if not case.id or not case.question:
            raise ValueError(f"{path}:{line_number}: id and question are required")
        if case.answerable and not case.reference:
            raise ValueError(f"{path}:{line_number}: answerable case has no gold_text")
        if categories and case.category not in categories:
            continue
        cases.append(case)
        if limit > 0 and len(cases) >= limit:
            break
    if not cases:
        raise ValueError("no benchmark cases selected")
    return cases


def block_id(evidence: dict[str, Any]) -> str | None:
    value = evidence.get("block_id")
    if value:
        return str(value)
    fragment = urlsplit(str(evidence.get("source") or "")).fragment
    return fragment or None


def first_gold_rank(case: BenchmarkCase, chunks: list[dict[str, Any]]) -> int | None:
    for rank, chunk in enumerate(chunks, 1):
        if block_id(chunk) in case.gold_block_ids:
            return rank
    return None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def metric_value(result: Any) -> float:
    value = float(result.value)
    if not math.isfinite(value):
        raise ValueError(f"Ragas returned non-finite score: {value}")
    return value


async def retrieve(
    client: httpx.AsyncClient,
    case: BenchmarkCase,
    *,
    top_k: int,
    enable_rerank: bool,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    response = await client.post(
        "/retrieve",
        json={
            "query": case.question,
            "mode": "text",
            "chunk_top_k": top_k,
            "enable_rerank": enable_rerank,
        },
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise TypeError(f"{case.id}: response chunks must be a list")
    return chunks, latency_ms


async def run(args: argparse.Namespace) -> int:
    ragas_home = _ROOT / "benchmarks" / "private" / ".ragas-home"
    ragas_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(ragas_home)
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"

    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall
    except ImportError as err:
        raise RuntimeError(
            "Ragas evaluation dependencies are missing; run "
            "`uv sync --extra eval` in the minirag project"
        ) from err

    cases = load_cases(
        args.dataset,
        categories=set(args.category),
        limit=args.limit,
    )
    settings = load_settings(args.config)
    judge_client = AsyncOpenAI(
        api_key=settings.chat.api_key,
        base_url=settings.chat.base_url,
        timeout=args.judge_timeout,
    )
    judge_llm = llm_factory(settings.chat.model, client=judge_client)
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
    ) as retrieval_client:
        for index, case in enumerate(cases, 1):
            row: dict[str, Any] = {
                "id": case.id,
                "question": case.question,
                "answerable": case.answerable,
                "category": case.category,
                "difficulty": case.difficulty,
            }
            try:
                chunks, latency_ms = await retrieve(
                    retrieval_client,
                    case,
                    top_k=args.top_k,
                    enable_rerank=not args.disable_rerank,
                )
                row["latency_ms"] = latency_ms
                row["retrieved"] = chunks
                rank = first_gold_rank(case, chunks)
                row["hit_at_k"] = rank is not None if case.answerable else None
                row["first_gold_rank"] = rank
                row["reciprocal_rank"] = 1.0 / rank if rank else 0.0
                row["unanswerable_false_positive"] = (
                    bool(chunks) if not case.answerable else None
                )

                if case.answerable and not chunks:
                    row["ragas_context_precision"] = 0.0
                    row["ragas_context_recall"] = 0.0
                elif case.answerable:
                    contexts = [str(chunk.get("text") or "") for chunk in chunks]
                    try:
                        precision_result = await context_precision.ascore(
                            user_input=case.question,
                            reference=case.reference,
                            retrieved_contexts=contexts,
                        )
                        recall_result = await context_recall.ascore(
                            user_input=case.question,
                            reference=case.reference,
                            retrieved_contexts=contexts,
                        )
                        row["ragas_context_precision"] = metric_value(precision_result)
                        row["ragas_context_recall"] = metric_value(recall_result)
                    except Exception as err:  # noqa: BLE001 - preserve other case results
                        row["ragas_error"] = f"{type(err).__name__}: {err}"
            except Exception as err:  # noqa: BLE001 - preserve partial experiment output
                row["retrieval_error"] = f"{type(err).__name__}: {err}"
            results.append(row)

            precision = row.get("ragas_context_precision")
            recall = row.get("ragas_context_recall")
            suffix = (
                f" precision={precision:.3f} recall={recall:.3f}"
                if isinstance(precision, float) and isinstance(recall, float)
                else ""
            )
            print(f"[{index}/{len(cases)}] {case.id}{suffix}")

    await judge_client.close()

    answerable_rows = [row for row in results if row["answerable"]]
    unanswerable_rows = [row for row in results if not row["answerable"]]
    latencies = [
        float(row["latency_ms"])
        for row in results
        if isinstance(row.get("latency_ms"), (int, float))
    ]
    precision_scores = [
        float(row["ragas_context_precision"])
        for row in answerable_rows
        if isinstance(row.get("ragas_context_precision"), (int, float))
    ]
    recall_scores = [
        float(row["ragas_context_recall"])
        for row in answerable_rows
        if isinstance(row.get("ragas_context_recall"), (int, float))
    ]
    hit_rows = [
        row
        for row in answerable_rows
        if not row.get("retrieval_error")
    ]
    false_positive_rows = [
        row
        for row in unanswerable_rows
        if row.get("unanswerable_false_positive") is True
    ]

    summary = {
        "cases": len(results),
        "answerable_cases": len(answerable_rows),
        "unanswerable_cases": len(unanswerable_rows),
        "retrieval_errors": sum("retrieval_error" in row for row in results),
        "ragas_errors": sum("ragas_error" in row for row in results),
        f"hit_at_{args.top_k}": mean_or_none(
            [1.0 if row.get("hit_at_k") else 0.0 for row in hit_rows]
        ),
        "mrr": mean_or_none(
            [float(row.get("reciprocal_rank") or 0.0) for row in hit_rows]
        ),
        "ragas_context_precision": mean_or_none(precision_scores),
        "ragas_context_recall": mean_or_none(recall_scores),
        "unanswerable_fpr": (
            len(false_positive_rows) / len(unanswerable_rows)
            if unanswerable_rows
            else None
        ),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
    }
    report = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": str(args.dataset),
            "base_url": args.base_url,
            "top_k": args.top_k,
            "rerank_enabled": not args.disable_rerank,
            "ragas_version": importlib.metadata.version("ragas"),
            "judge_provider": settings.chat.provider,
            "judge_model": settings.chat.model,
            "judge_timeout_seconds": args.judge_timeout,
            "ragas_tracking_disabled": True,
        },
        "summary": summary,
        "cases": results,
    }

    output = args.output or (
        _ROOT
        / "benchmarks"
        / "private"
        / "results"
        / f"ragas-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report={output}")

    return 1 if summary["retrieval_errors"] or summary["ragas_errors"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MiniRAG retrieval with Ragas context metrics",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=_ROOT / "benchmarks" / "private" / "cloudwan_retrieval.jsonl",
    )
    parser.add_argument("--base-url", default="http://localhost:8090")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--judge-timeout", type=float, default=300.0)
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.limit < 0:
        parser.error("--limit must not be negative")
    if args.timeout <= 0 or args.judge_timeout <= 0:
        parser.error("--timeout and --judge-timeout must be positive")
    return args


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
