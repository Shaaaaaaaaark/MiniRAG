"""MiniRAG 门面：装配存储/模型/索引/检索，提供索引与检索两个入口。

用法：
    rag = MiniRAG()          # 读取 config.yaml；不存在时回退到 config.example.yaml
    await rag.startup()
    await rag.index(DocumentInput(source="corpus/x.md"))
    result = await rag.retrieve("BGP 中断如何处理", QueryParam(mode="mix"))
    await rag.shutdown()
"""
from __future__ import annotations

from minirag.config import Settings, get_settings
from minirag.core.index import Indexer
from minirag.core.retrieve import Retriever
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
        self.indexer = Indexer(self.settings, self.models, self.pg, self.milvus)
        self.retriever = Retriever(self.settings.retrieval, self.models, self.pg, self.milvus)

    async def startup(self) -> None:
        await self.pg.connect()
        await self.milvus.connect()

    async def shutdown(self) -> None:
        await self.pg.close()

    async def index(self, doc_input: DocumentInput, gleaning: bool = False) -> IndexReport:
        return await self.indexer.index_document(doc_input, gleaning=gleaning)

    async def retrieve(self, query: str, param: QueryParam | None = None) -> RetrievalResult:
        return await self.retriever.retrieve(query, param)
