"""LightRAG 内核 Prompt 模板：实体关系抽取、Gleaning、双层关键词。

约束：全部要求模型输出 JSON，禁止输出思维链或额外解释文本。
"""
from __future__ import annotations

from minirag.schemas import Chunk, ExtractedGraph, Message

_ENTITY_TYPES = (
    "Service, CloudResource, Region, Alert, Metric, ErrorCode, Tool, Procedure, Constraint"
)

_EXTRACT_SYSTEM = f"""你是云网络领域的知识抽取器。从给定文本中抽取实体与关系。
实体 type 只能取以下之一：{_ENTITY_TYPES}。
关系需给出 src、dst（实体名）、keywords、description、weight(0~1)。
只输出符合 schema 的 JSON，不要输出任何解释或思考过程。"""

# 双层关键词：high_level 面向主题/概念/意图，low_level 面向具体实体/专名/错误码
_KEYWORDS_SYSTEM = """你是检索关键词抽取器。从用户问题中抽取两层关键词：
high_level：主题、概念、场景、意图类关键词（用于检索"关系"）；
low_level：具体实体、资源名、错误码、专有名词类关键词（用于检索"实体"）。
只输出 JSON，不要解释。对于过于简单或无意义的问题，返回空数组。"""


def build_extract_messages(chunk: Chunk) -> list[Message]:
    return [
        Message(role="system", content=_EXTRACT_SYSTEM),
        Message(role="user", content=f"标题路径：{chunk.heading_path}\n正文：\n{chunk.content}"),
    ]


def build_gleaning_messages(chunk: Chunk, first: ExtractedGraph) -> list[Message]:
    seen = ", ".join(e.name for e in first.entities) or "（无）"
    return [
        Message(role="system", content=_EXTRACT_SYSTEM),
        Message(
            role="user",
            content=(
                f"已抽取实体：{seen}\n"
                f"请从下面文本补充遗漏的实体与关系，不要重复已有实体。\n"
                f"标题路径：{chunk.heading_path}\n正文：\n{chunk.content}"
            ),
        ),
    ]


def build_keywords_messages(query: str) -> list[Message]:
    return [
        Message(role="system", content=_KEYWORDS_SYSTEM),
        Message(role="user", content=query),
    ]
