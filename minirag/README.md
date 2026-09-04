# MiniRAG

面向技术文档的轻量检索服务。系统只返回结构化证据，不负责生成答案。

## 检索链路

```text
飞书/Markdown
  -> 结构化 Block
  -> Parent-Child 切片
  -> Milvus Dense + BM25
  -> RRF
  -> PostgreSQL 回填 Parent
  -> Rerank（失败时保留 RRF 顺序）
  -> Evidence
```

核心数据模型：

| 模型 | 职责 |
|---|---|
| Document | 保存稳定来源、revision 和处理状态 |
| Parent Chunk | 保留完整章节，作为返回上下文 |
| Child Chunk | 小粒度检索单元，写入 Milvus |
| Evidence | 返回正文、分数、标题路径、revision 和原文锚点 |

Milvus 负责 Dense/BM25 混合召回；PostgreSQL 保存文档、Child 与 Parent
上下文。查询命中 Child 后，从 PostgreSQL 回填并去重 Parent。

## 目录

```text
minirag/
  config.py              配置加载
  schemas.py             文档与检索契约
  models/                Embedding/Rerank 适配器
  storage/
    milvus_store.py      Dense + BM25 + RRF
    pg_store.py          Document/Parent/Child 元数据
  core/
    chunker.py           Markdown 解析与 Parent-Child 切片
    tokenizer.py         token 计数与窗口切分
    index.py             索引编排
    ranking.py           Rerank 映射与 token 预算
    retrieve.py          检索编排
  integrations/
    feishu.py            飞书读取与规范化
    feishu_sync.py       目录增量同步
    feishu_jit.py        单文档实时检索
  api/server.py          FastAPI 接口
```

## 配置

```bash
cd minirag
cp config.example.yaml config.yaml
uv sync --extra dev
```

真实 Key 和私有 endpoint 只放在 `config.yaml` 或环境变量中。`config.yaml`
已被 Git 忽略，也可以通过 `MINIRAG_CONFIG` 指定其他配置文件。

在线检索只调用 Embedding 和 Rerank。配置中的可选 `chat` 仅供
`benchmarks/run_ragas.py` 作为 Judge 使用。

## 运行

启动依赖：

```bash
docker compose up -d standalone postgres attu
```

启动 API：

```bash
cd minirag
uv run uvicorn minirag.api.server:app --port 8090
```

Swagger：<http://localhost:8090/docs>

索引本地文档：

```bash
curl -X POST http://localhost:8090/documents \
  -H 'Content-Type: application/json' \
  -d '{"source":"/path/to/document.md"}'
```

检索：

```bash
curl -X POST http://localhost:8090/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"CreateGlobalNetwork 的 ClientToken 有什么约束？",
    "mode":"text",
    "chunk_top_k":5,
    "enable_rerank":true
  }'
```

`mode=text`、`top_k` 和 `chunk_top_k` 为 HTTP 兼容字段；内部只有一条文本混合
检索链路，`chunk_top_k` 优先于 `top_k`。

## Python API

```python
from minirag import DocumentInput, MiniRAG, QueryParam

rag = MiniRAG()
await rag.startup()
try:
    await rag.index(DocumentInput(source="/path/to/document.md"))
    result = await rag.retrieve(
        "BGP 中断如何处理",
        QueryParam(top_k=8),
    )
finally:
    await rag.shutdown()
```

## 飞书

增量同步目录：

```bash
uv run python scripts/sync_feishu.py \
  --folder "https://example.feishu.cn/drive/folder/..." \
  --index
```

同步产物默认写入 `corpus/feishu/`。脚本通过
`doc_token + revision_id + modified_time` 判断是否需要重新读取；只有指定
`--prune` 才删除已经移出目录的本地文件和索引。

实时读取单篇文档：

```bash
curl -X POST http://localhost:8090/feishu/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "source_url":"https://example.feishu.cn/docx/xxx",
    "query":"灰度流程是什么？",
    "top_k":5
  }'
```

JIT 依赖本机已登录的 `lark-cli`。当前服务没有文档级 ACL，只适合单用户或固定
授权目录。

## 测试与评测

```bash
uv run --extra dev python -m pytest -q
python scripts/smoke.py retrieve "BGP 中断如何处理" --top-k 8
```

固定测试集、确定性指标和 Ragas 方案见
[benchmarks/README.md](../benchmarks/README.md)。

LangChain 对照实现位于
[minirag-langchain](../minirag-langchain/README.md)，使用独立 collection，
可以通过同一套 benchmark 比较检索效果。

## 当前边界

- 索引接口仍为同步执行，尚未拆分独立 Worker。
- 重索引采用替换策略，尚未实现 active revision 原子切换。
- 没有文档级 ACL 和无答案阈值。
- 不包含答案生成和 Agent 编排。
