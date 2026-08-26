"""实体关系抽取：LLM + JSON Schema，含最多一次 Gleaning 补充抽取。

对齐官方 extract_entities 的精简版：初始抽取 1 次 + gleaning 1 次（可关），
两次结果按规范化实体名/关系去重合并。抽取禁用 thinking 深度思考以提速。
"""
from __future__ import annotations

from minirag.core import prompts
from minirag.core.fusion import normalize_name
from minirag.models.base import ChatModel
from minirag.schemas import Chunk, Entity, ExtractedGraph, Relation


def _merge_graph(first: ExtractedGraph, more: ExtractedGraph) -> ExtractedGraph:
    entities: dict[str, Entity] = {}
    for e in list(first.entities) + list(more.entities):
        entities.setdefault(normalize_name(e.name), e)
    relations: dict[tuple[str, str], Relation] = {}
    for r in list(first.relations) + list(more.relations):
        relations.setdefault((normalize_name(r.src), normalize_name(r.dst)), r)
    return ExtractedGraph(entities=list(entities.values()), relations=list(relations.values()))


async def extract(chunk: Chunk, chat: ChatModel, gleaning: bool = False) -> ExtractedGraph:
    """从单个 chunk 抽取实体与关系。gleaning=True 时执行一次补充抽取。"""
    graph = await chat.generate(prompts.build_extract_messages(chunk), ExtractedGraph, thinking=False)
    if not gleaning:
        return graph
    more = await chat.generate(
        prompts.build_gleaning_messages(chunk, graph), ExtractedGraph, thinking=False
    )
    return _merge_graph(graph, more)
