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
