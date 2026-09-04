# MiniRAG LangChain

使用 LangChain 实现与 MiniRAG 相同的文本检索链路，用于比较框架封装与自定义
实现。它使用独立的虚拟环境和 Milvus collection，不修改主实现的数据。

## 链路

```text
规范 Markdown
  -> MarkdownHeaderTextSplitter
  -> RecursiveCharacterTextSplitter（Parent / Child）
  -> OpenAIEmbeddings 兼容接口
  -> LangChain Milvus VectorStore
       Dense COSINE + Milvus BM25 + RRF
  -> Parent 回填与去重
  -> DashScope Rerank（失败时保留 RRF 顺序）
  -> Evidence
```

LangChain 提供 `Document`、Splitter、Embedding 和 Milvus 集成。飞书元数据解析、
Parent-Child 关联、Rerank 降级和 Evidence 契约仍由应用实现。

## 安装

```bash
cd minirag-langchain
uv sync --extra dev
cp config.example.yaml config.yaml
```

也可以复用主实现的私有配置：

```bash
export MINIRAG_LANGCHAIN_CONFIG=../minirag/config.yaml
```

默认使用独立 collection `langchain_chunks`。

## 索引与检索

全量重建飞书规范语料：

```bash
uv run langchain-rag index ../corpus/feishu/docs
```

检索：

```bash
uv run langchain-rag retrieve \
  "CreateGlobalNetwork 的 ClientToken 有什么约束？" \
  --top-k 5
```

查看 RRF 原始排序：

```bash
uv run langchain-rag retrieve \
  "CreateGlobalNetwork 的 ClientToken 有什么约束？" \
  --top-k 5 \
  --disable-rerank
```

## HTTP API

```bash
uv run langchain-rag serve --port 8091
```

Swagger：<http://localhost:8091/docs>

```bash
curl -X POST http://localhost:8091/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"CreateGlobalNetwork 的 ClientToken 有什么约束？",
    "mode":"text",
    "chunk_top_k":5,
    "enable_rerank":true
  }'
```

请求和响应兼容 benchmark 使用的 MiniRAG 子集：

```bash
uv run --project minirag --extra eval \
  python benchmarks/run_ragas.py \
  --base-url http://localhost:8091 \
  --limit 3 \
  --top-k 5
```

## 测试

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev ruff check src tests
```

## 边界

- 只读取已经规范化的 Markdown，不负责飞书同步和 JIT。
- Parent 内容随 Child metadata 写入 Milvus，不使用 PostgreSQL。
- 索引命令执行全量重建，不实现 revision 生命周期。
- 不包含 ACL、无答案阈值、答案生成和 Agent。
