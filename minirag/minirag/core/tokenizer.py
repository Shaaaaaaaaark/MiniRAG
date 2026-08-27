"""Tokenizer 封装：统一用 tiktoken cl100k_base 计数，供切分与预算截断使用。"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def split_by_tokens(text: str, size: int) -> list[str]:
    """将文本按 token 硬切为每片不超过 size 的片段（无自然边界时兜底）。"""
    enc = _encoder()
    ids = enc.encode(text)
    return [enc.decode(ids[i : i + size]) for i in range(0, len(ids), size)]


def split_by_token_windows(text: str, size: int, overlap: int = 0) -> list[str]:
    """按 token 滑窗切分；相邻窗口保留 overlap 个 token。"""
    if size <= 0:
        raise ValueError("size 必须大于 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")
    enc = _encoder()
    ids = enc.encode(text)
    if not ids:
        return []
    step = size - overlap
    windows: list[str] = []
    for start in range(0, len(ids), step):
        windows.append(enc.decode(ids[start : start + size]))
        if start + size >= len(ids):
            break
    return windows
