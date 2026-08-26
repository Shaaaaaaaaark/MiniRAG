"""LightRAG 内核：解析、切分、抽取、融合、索引与检索。

为避免导入副作用（拉起 openai/pymilvus 等重依赖），本包不做便捷 re-export，
请从具体子模块显式导入，例如：
    from minirag.core.index import Indexer
    from minirag.core.retrieve import Retriever
"""
