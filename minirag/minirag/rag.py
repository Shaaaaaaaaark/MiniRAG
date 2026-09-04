"""MiniRAG 门面：装配模型、存储、索引与检索。

用法：
    rag = MiniRAG()          # 读取 config.yaml，或使用 MINIRAG_CONFIG 指定路径
    await rag.startup()
    await rag.index(DocumentInput(source="corpus/x.md"))
    result = await rag.retrieve("BGP 中断如何处理", QueryParam(top_k=8))
    await rag.shutdown()
"""
from __future__ import annotations

from minirag.config import Settings, get_settings
from minirag.core.index import Indexer
from minirag.core.retrieve import Retriever
from minirag.integrations.feishu import FeishuCliClient
from minirag.integrations.feishu_jit import FeishuJitRetriever
from minirag.models.factory import ModelBundle
from minirag.schemas import DocumentInput, IndexReport, QueryParam, RetrievalResult
from minirag.storage.milvus_store import MilvusStore
from minirag.storage.pg_store import PgStore


class MiniRAG:
    """进程级门面：集中持有所有运行期组件。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.models = ModelBundle(self.settings)
        self.pg = PgStore(self.settings.postgres)
        self.milvus = MilvusStore(self.settings.milvus, dim=self.models.embedding.dimensions)
        self.indexer = Indexer(self.models, self.pg, self.milvus)
        self.retriever = Retriever(self.settings.retrieval, self.models, self.pg, self.milvus)
        self.feishu_client: FeishuCliClient | None = None
        self.feishu_jit: FeishuJitRetriever | None = None
        if self.settings.feishu.enabled:
            self.feishu_client = FeishuCliClient(
                cli_path=self.settings.feishu.cli_path,
                identity=self.settings.feishu.identity,
                timeout_seconds=self.settings.feishu.timeout_seconds,
            )
            self.feishu_jit = FeishuJitRetriever(self.feishu_client, self.models)

    async def startup(self) -> None:
        await self.pg.connect()
        await self.milvus.connect()

    async def shutdown(self) -> None:
        try:
            await self.pg.close()
        finally:
            await self.milvus.close()

    async def index(self, doc_input: DocumentInput) -> IndexReport:
        return await self.indexer.index_document(doc_input)

    async def delete_document(self, source: str, source_id: str | None = None) -> None:
        await self.indexer.delete_document(source, source_id)

    async def retrieve(self, query: str, param: QueryParam | None = None) -> RetrievalResult:
        return await self.retriever.retrieve(query, param)

    async def retrieve_feishu(
        self,
        source_url: str,
        query: str,
        *,
        top_k: int = 8,
        enable_rerank: bool = True,
    ) -> RetrievalResult:
        if self.feishu_jit is None:
            raise RuntimeError(
                "飞书 JIT 未启用，请在 config.yaml 设置 feishu.enabled=true"
            )
        return await self.feishu_jit.retrieve(
            source_url,
            query,
            top_k=top_k,
            enable_rerank=enable_rerank,
        )
