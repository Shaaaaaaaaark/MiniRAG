"""数据契约：消息、图谱实体/关系、分块、证据、检索参数与结果。

Evidence 一旦生成其 evidence_id 不可变，最终结论必须引用其 evidence_id。
"""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

EvidenceKind = Literal["chunk", "entity", "relation"]
QueryMode = Literal["text", "local", "global", "hybrid", "mix", "naive"]

# 云网络固定实体类型
EntityType = Literal[
    "Service",
    "CloudResource",
    "Region",
    "Alert",
    "Metric",
    "ErrorCode",
    "Tool",
    "Procedure",
    "Constraint",
]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


def make_evidence_id(kind: str, ref_id: str) -> str:
    """由 (kind, ref_id) 生成稳定、可复现的证据 ID。"""
    return "e_" + hashlib.sha1(f"{kind}:{ref_id}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 索引期：文档 / 分块 / 图谱
# ---------------------------------------------------------------------------

BlockType = Literal["heading", "paragraph", "table"]
ChunkingStrategy = Literal["header_token", "parent_child"]


class Block(BaseModel):
    type: BlockType
    text: str
    level: int | None = None
    block_id: str | None = None


class DocumentInput(BaseModel):
    source: str
    source_id: str | None = None
    revision: str | None = None
    title: str | None = None
    text: str | None = None  # 直接传文本时使用（否则按 source 路径读取）
    blocks: list[Block] | None = None
    chunking_strategy: ChunkingStrategy | None = None


class ParsedDocument(BaseModel):
    document_id: str
    source: str
    source_id: str | None = None
    title: str | None = None
    revision: str | None = None
    blocks: list[Block] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    document_id: str
    ord: int
    heading_path: str
    content: str
    token_count: int
    parent_id: str | None = None
    parent_content: str | None = None
    block_id: str | None = None


class Entity(BaseModel):
    name: str
    type: EntityType
    description: str = ""

    @field_validator("description", mode="before")
    @classmethod
    def _desc_default(cls, v: object) -> str:
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)


class Relation(BaseModel):
    src: str
    dst: str
    keywords: str
    description: str = ""
    weight: float = 1.0

    @field_validator("keywords", "description", mode="before")
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        return v if isinstance(v, str) else str(v)


class ExtractedGraph(BaseModel):
    """单个 chunk 抽取出的实体与关系（LLM 结构化输出目标）。"""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class IndexReport(BaseModel):
    document_id: str
    chunks: int = 0
    entities: int = 0
    relations: int = 0


# ---------------------------------------------------------------------------
# 检索期：关键词 / 证据 / 参数 / 结果
# ---------------------------------------------------------------------------


class Keywords(BaseModel):
    """双层关键词：high_level 面向主题/关系，low_level 面向具体实体。"""

    high_level: list[str] = Field(default_factory=list)
    low_level: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """检索得到的单条证据。"""

    evidence_id: str
    kind: EvidenceKind
    ref_id: str
    text: str
    source: str
    heading_path: str | None = None
    score: float = 0.0
    parent_id: str | None = None
    block_id: str | None = None
    revision: str | None = None


class QueryParam(BaseModel):
    """检索参数（精简自官方 QueryParam）。"""

    mode: QueryMode = "text"
    top_k: int | None = None          # KG 实体/关系召回条数（None 用 config 默认）
    chunk_top_k: int | None = None    # chunk 召回条数
    max_entity_tokens: int | None = None
    max_relation_tokens: int | None = None
    max_total_tokens: int | None = None
    enable_rerank: bool = True


class RetrievalResult(BaseModel):
    """一次检索的结构化产出（只检索、不生成）。"""

    keywords: Keywords = Field(default_factory=Keywords)
    entities: list[Evidence] = Field(default_factory=list)
    relationships: list[Evidence] = Field(default_factory=list)
    chunks: list[Evidence] = Field(default_factory=list)

    @property
    def all_evidences(self) -> list[Evidence]:
        return [*self.entities, *self.relationships, *self.chunks]
