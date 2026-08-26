"""模型工厂：按配置 provider 装配 chat/embedding/rerank 实例。

三个适配器均基于 OpenAI 兼容协议（rerank 走 DashScope 原生 REST），
provider 同时接受具体厂商名与通用 `openai`；新增厂商在此登记。
"""
from __future__ import annotations

from minirag.config import ModelCfg, Settings
from minirag.models.ark_chat import ArkChatModel
from minirag.models.bailian_embed import BailianEmbeddingModel
from minirag.models.bailian_rerank import BailianRerankModel
from minirag.models.base import ChatModel, EmbeddingModel, RerankModel

_CHAT_PROVIDERS = {"ark", "openai", "bailian"}
_EMBEDDING_PROVIDERS = {"bailian", "openai"}
_RERANK_PROVIDERS = {"bailian", "openai"}


def build_chat_model(cfg: ModelCfg) -> ChatModel:
    if cfg.provider in _CHAT_PROVIDERS:
        return ArkChatModel(cfg)
    raise ValueError(f"不支持的 chat provider: {cfg.provider}")


def build_embedding_model(cfg: ModelCfg) -> EmbeddingModel:
    if cfg.provider in _EMBEDDING_PROVIDERS:
        return BailianEmbeddingModel(cfg)
    raise ValueError(f"不支持的 embedding provider: {cfg.provider}")


def build_rerank_model(cfg: ModelCfg) -> RerankModel:
    if cfg.provider in _RERANK_PROVIDERS:
        return BailianRerankModel(cfg)
    raise ValueError(f"不支持的 rerank provider: {cfg.provider}")


class ModelBundle:
    """三件套模型集合，供索引与检索注入。"""

    def __init__(self, settings: Settings) -> None:
        self.chat = build_chat_model(settings.chat)
        self.embedding = build_embedding_model(settings.embedding)
        self.rerank = build_rerank_model(settings.rerank)
