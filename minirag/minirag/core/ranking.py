"""Rerank 结果映射与 token 预算截断。"""
from __future__ import annotations

from minirag.core.tokenizer import count_tokens
from minirag.models.base import RerankResult
from minirag.schemas import Evidence


def apply_rerank(
    evidences: list[Evidence],
    rerank_results: list[RerankResult],
) -> list[Evidence]:
    """按 Rerank 返回的顺序与分数重排证据。"""
    ranked: list[Evidence] = []
    for result in rerank_results:
        if 0 <= result.index < len(evidences):
            evidence = evidences[result.index]
            evidence.score = result.score
            ranked.append(evidence)
    return ranked


def take_within_budget(
    evidences: list[Evidence],
    token_budget: int,
) -> list[Evidence]:
    """按顺序选取不超过 token 预算的证据。"""
    selected: list[Evidence] = []
    used = 0
    for evidence in evidences:
        cost = count_tokens(evidence.text)
        if used + cost <= token_budget:
            selected.append(evidence)
            used += cost
    return selected
