"""检索融合工具：RRF 融合、rerank 应用、分层 token 预算截断（确定性纯函数）。"""
from __future__ import annotations

from minirag.core.tokenizer import count_tokens
from minirag.models.base import RerankResult
from minirag.schemas import Evidence


def rrf_merge(ranked_lists: list[list[Evidence]], k: int = 60) -> list[Evidence]:
    """Reciprocal Rank Fusion：合并多路召回，按 RRF 分数降序去重。

    同一 evidence_id 在多路出现时分数累加：score += 1 / (k + rank)。
    """
    scores: dict[str, float] = {}
    best: dict[str, Evidence] = {}
    for evidences in ranked_lists:
        for rank, ev in enumerate(evidences):
            scores[ev.evidence_id] = scores.get(ev.evidence_id, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(ev.evidence_id, ev)
    ordered = sorted(best.values(), key=lambda e: scores[e.evidence_id], reverse=True)
    for ev in ordered:
        ev.score = scores[ev.evidence_id]
    return ordered


def round_robin_merge(list_a: list[Evidence], list_b: list[Evidence]) -> list[Evidence]:
    """交替合并两路结果并按 evidence_id 去重（对应官方 hybrid 的 round-robin）。"""
    out: list[Evidence] = []
    seen: set[str] = set()
    for i in range(max(len(list_a), len(list_b))):
        for src in (list_a, list_b):
            if i < len(src):
                ev = src[i]
                if ev.evidence_id not in seen:
                    seen.add(ev.evidence_id)
                    out.append(ev)
    return out


def apply_rerank(evidences: list[Evidence], rerank_results: list[RerankResult]) -> list[Evidence]:
    """按 rerank 返回的顺序与分数重排 evidences。"""
    out: list[Evidence] = []
    for r in rerank_results:
        if 0 <= r.index < len(evidences):
            ev = evidences[r.index]
            ev.score = r.score
            out.append(ev)
    return out


def take_within_budget(evidences: list[Evidence], token_budget: int) -> list[Evidence]:
    """按顺序累计 token，不超过预算；单条超预算则跳过。"""
    selected: list[Evidence] = []
    used = 0
    for ev in evidences:
        cost = count_tokens(ev.text)
        if used + cost > token_budget:
            continue
        selected.append(ev)
        used += cost
    return selected
