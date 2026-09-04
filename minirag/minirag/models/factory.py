"""模型工厂：装配 Embedding 与 Rerank 适配器。"""
from __future__ import annotations

from minirag.config import ModelCfg, Settings
from minirag.models.bailian_embed import BailianEmbeddingModel
from minirag.models.bailian_rerank import BailianRerankModel
from minirag.models.base import EmbeddingModel, RerankModel

_EMBEDDING_PROVIDERS = {"bailian", "openai"}
_RERANK_PROVIDERS = {"bailian"}


def build_embedding_model(cfg: ModelCfg) -> EmbeddingModel:
    if cfg.provider in _EMBEDDING_PROVIDERS:
        return BailianEmbeddingModel(cfg)
    raise ValueError(f"不支持的 embedding provider: {cfg.provider}")


def build_rerank_model(cfg: ModelCfg) -> RerankModel:
    if cfg.provider in _RERANK_PROVIDERS:
        return BailianRerankModel(cfg)
    raise ValueError(f"不支持的 rerank provider: {cfg.provider}")


class ModelBundle:
    """运行期模型集合。"""

    def __init__(self, settings: Settings) -> None:
        self.embedding = build_embedding_model(settings.embedding)
        self.rerank = build_rerank_model(settings.rerank)
