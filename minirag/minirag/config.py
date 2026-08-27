"""配置加载：读取 config.yaml，并将 ${ENV_VAR} 占位符替换为环境变量值。

所有运行期参数集中在此，供模型层、存储层、索引与检索层注入使用。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


class ModelCfg(BaseModel):
    """单个模型（chat/embedding/rerank）的连接配置。"""

    provider: str
    model: str
    base_url: str
    api_key: str
    dimensions: int | None = None
    temperature: float = 0.0
    timeout: float | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    # 透传给底层 chat.completions.create 的额外参数（如方舟 thinking 深度思考）
    extra_body: dict | None = None

    @property
    def effective_timeout(self) -> float:
        return self.timeout if self.timeout is not None else self.timeout_seconds


class MilvusCfg(BaseModel):
    uri: str
    db: str = "oncall"


class PostgresCfg(BaseModel):
    dsn: str


class FeishuCfg(BaseModel):
    enabled: bool = False
    cli_path: str = "lark-cli"
    identity: str = "user"
    timeout_seconds: float = 120.0


class RetrievalCfg(BaseModel):
    """检索固定参数（分层召回 + 分层 token 预算）。"""

    entity_topk: int = 10
    relation_topk: int = 10
    dense_topk: int = 20
    bm25_topk: int = 20
    rerank_topk: int = 8
    rrf_k: int = 60
    chunk_top_k: int = 8
    max_entity_tokens: int = 4000
    max_relation_tokens: int = 4000
    max_total_tokens: int = 12000


class Settings(BaseModel):
    chat: ModelCfg
    embedding: ModelCfg
    rerank: ModelCfg
    milvus: MilvusCfg
    postgres: PostgresCfg
    feishu: FeishuCfg = Field(default_factory=FeishuCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    chunker: str = "header_token"
    graph_enabled: bool = False


def _expand_env(value: Any) -> Any:
    """递归地将字符串中的 ${VAR} 或 ${VAR:-default} 替换为环境变量值。"""

    def _sub(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        return default if default is not None else m.group(0)

    if isinstance(value, str):
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """从 YAML 加载配置。

    默认读取包同级目录的 config.yaml；公开仓库只提交 config.example.yaml 模板。
    """
    import yaml
    from dotenv import load_dotenv

    if config_path:
        path = Path(config_path)
    else:
        config_dir = Path(__file__).resolve().parents[1]
        path = config_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"MiniRAG 配置文件不存在: {path}。请复制 config.example.yaml 为 config.yaml，"
            "或通过 MINIRAG_CONFIG 指定配置文件。"
        )
    load_dotenv(path.parent / ".env", override=False)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(_expand_env(raw))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings(os.environ.get("MINIRAG_CONFIG") or None)
