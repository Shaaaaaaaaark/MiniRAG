"""图谱融合工具：实体/关系的规范化、ID 生成与确定性合并。

同名实体合并描述与来源；同 (src,dst) 关系聚合 keywords/描述、累加 weight。
不做 LLM 描述归纳（精简版取舍）。
"""
from __future__ import annotations

import hashlib

from minirag.schemas import Entity, Relation


def normalize_name(name: str) -> str:
    """实体名规范化：去空白、统一小写、全角转半角，用于生成稳定 ID。"""
    text = name.strip().lower()
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out).strip()


def entity_id(name: str) -> str:
    return "ent_" + hashlib.sha1(normalize_name(name).encode("utf-8")).hexdigest()[:16]


def relation_id(src: str, dst: str) -> str:
    """关系 ID：按规范化名排序后生成，保证无向边 (a,b)==(b,a)。"""
    a, b = sorted([normalize_name(src), normalize_name(dst)])
    return "rel_" + hashlib.sha1(f"{a}->{b}".encode("utf-8")).hexdigest()[:16]


def canonical_edge_ids(src: str, dst: str) -> tuple[str, str]:
    """返回规范化后的 (src_node_id, tgt_node_id)，src_id <= tgt_id。

    用 Python 排序而非 SQL LEAST/GREATEST，避免中文在非 C collation 下分叉。
    """
    sid, tid = entity_id(src), entity_id(dst)
    return (sid, tid) if sid <= tid else (tid, sid)


def _merge_descriptions(old: str, new: str) -> str:
    parts: list[str] = []
    for seg in (old, new):
        seg = seg.strip()
        if seg and seg not in parts:
            parts.append(seg)
    return "\n".join(parts)


def merge_entity(existing: Entity | None, new: Entity) -> Entity:
    if existing is None:
        return new
    return Entity(
        name=existing.name,
        type=existing.type,
        description=_merge_descriptions(existing.description, new.description),
    )


def merge_relation(existing: Relation | None, new: Relation) -> Relation:
    if existing is None:
        return new
    return Relation(
        src=existing.src,
        dst=existing.dst,
        keywords=_merge_descriptions(existing.keywords, new.keywords),
        description=_merge_descriptions(existing.description, new.description),
        weight=existing.weight + new.weight,
    )
