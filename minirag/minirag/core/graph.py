"""图扩展：把向量命中的实体/关系，沿 PG 图表做「一跳邻接」扩展。

对应官方 _get_node_data / _get_edge_data 里对图存储的调用（一跳 + 度数）：
- expand_entities：实体命中 → 取邻边（关系证据）+ 邻边来源 chunk
- relation_source_chunks：关系命中 → 取其来源 chunk
边/关系证据的 evidence_id 与索引期一致（rel_ 前缀经 make_evidence_id 包装）。
"""
from __future__ import annotations

from minirag.schemas import Evidence, make_evidence_id
from minirag.storage.pg_store import PgStore


def _edge_to_evidence(edge: dict) -> Evidence:
    rid = edge["id"]
    text = f"{edge.get('keywords', '')}: {edge.get('description', '')}".strip(": ")
    return Evidence(
        evidence_id=make_evidence_id("relation", rid),
        kind="relation",
        ref_id=rid,
        text=text,
        source="graph_edges",
        score=float(edge.get("weight", 0.0)),
    )


async def expand_entities_to_relations(
    entity_evidences: list[Evidence], pg: PgStore
) -> list[Evidence]:
    """由命中实体沿图取邻边，按 (度数, weight) 降序，转为关系证据。"""
    if not entity_evidences:
        return []
    node_ids = [e.ref_id for e in entity_evidences]
    degrees = await pg.node_degrees_batch(node_ids)

    seen_edges: dict[str, dict] = {}
    for e in entity_evidences:
        edges = await pg.get_node_edges(e.ref_id)
        for edge in edges:
            seen_edges.setdefault(edge["id"], edge)

    # rank = 两端度数之和，用于排序（对应官方 edge rank）
    def edge_rank(edge: dict) -> tuple[int, float]:
        r = degrees.get(edge["src_id"], 0) + degrees.get(edge["tgt_id"], 0)
        return (r, float(edge.get("weight", 0.0)))

    ordered = sorted(seen_edges.values(), key=edge_rank, reverse=True)
    return [_edge_to_evidence(edge) for edge in ordered]


async def expand_relations_to_entities(
    relation_evidences: list[Evidence], pg: PgStore
) -> list[Evidence]:
    """由命中关系取其两端实体节点，转为实体证据。"""
    if not relation_evidences:
        return []
    edge_ids = [e.ref_id for e in relation_evidences]
    edges = await pg.edges_by_ids(edge_ids)

    node_ids: list[str] = []
    for edge in edges:
        node_ids.extend([edge["src_id"], edge["tgt_id"]])
    nodes = await pg.nodes_by_ids(list(dict.fromkeys(node_ids)))

    out: list[Evidence] = []
    seen: set[str] = set()
    for nid, node in nodes.items():
        if nid in seen:
            continue
        seen.add(nid)
        text = f"{node.get('name', '')}: {node.get('description', '')}".strip(": ")
        out.append(
            Evidence(
                evidence_id=make_evidence_id("entity", nid),
                kind="entity",
                ref_id=nid,
                text=text,
                source="graph_nodes",
            )
        )
    return out


async def source_chunks_of_edges(
    relation_evidences: list[Evidence], pg: PgStore
) -> list[Evidence]:
    """取关系边的来源 chunk（图扩展落到文本层）。"""
    if not relation_evidences:
        return []
    edge_ids = [e.ref_id for e in relation_evidences]
    edges = await pg.edges_by_ids(edge_ids)
    chunk_ids: list[str] = []
    for edge in edges:
        chunk_ids.extend(edge.get("source_chunks") or [])
    return await pg.chunks_by_ids(list(dict.fromkeys(chunk_ids)))


async def source_chunks_of_entities(
    entity_evidences: list[Evidence], pg: PgStore
) -> list[Evidence]:
    """取实体节点的来源 chunk。"""
    if not entity_evidences:
        return []
    node_ids = [e.ref_id for e in entity_evidences]
    nodes = await pg.nodes_by_ids(node_ids)
    chunk_ids: list[str] = []
    for node in nodes.values():
        chunk_ids.extend(node.get("source_chunks") or [])
    return await pg.chunks_by_ids(list(dict.fromkeys(chunk_ids)))
