# MiniRAG — 精简轻量版 LightRAG 检索内核

从官方 LightRAG（v1.5.7）提取「双层图谱检索」算法内核，重写的窄实现。砍掉所有生产设施
（多存储后端、pipeline 并发、doc-status 恢复、sidecar、多解析器、WebUI、鉴权等），
只保留**索引 + 检索 + 多模式**，存储硬编码 **Milvus + PostgreSQL**，对外「检索即服务」。

## 核心算法

> 索引期用 LLM 把文档拆成「实体-关系图 + 文本 chunk」双层结构；查询期抽取双层关键词，
> 低层查实体、高层查关系，各自沿图扩展一跳，再与向量 chunk 融合、按分层预算截断，只返回证据。

检索模式（`QueryParam.mode`，默认 `mix`）：

| 模式 | 逻辑 |
|---|---|
| `local` | low 关键词 → 实体向量 → 图取邻边(关系) + 实体来源 chunk |
| `global` | high 关键词 → 关系向量 → 取两端实体 + 关系来源 chunk |
| `hybrid` | local + global，round-robin 合并 |
| `mix`（默认） | hybrid + query 向量召回 chunk |
| `naive` | 仅 query 向量召回 chunk（退化为传统 RAG） |

## 目录

```
minirag/
  config.py          配置加载（YAML + ${ENV}）
  schemas.py         数据契约 + QueryParam
  models/            模型抽象层 + 方舟/百炼适配器 + 工厂
  storage/
    milvus_store.py  向量三 Collection（chunk 走 dense+BM25 中文混合检索）
    pg_store.py      文档/分块元数据 + 图谱两表（graph_nodes/graph_edges，无需 AGE）
  core/
    chunker.py       解析(MD/TXT) + 标题层级/Token 切分
    tokenizer.py     tiktoken 计数
    prompts.py       抽取 / 双层关键词 prompt
    extract.py       实体关系抽取（LLM + 一次 gleaning）
    fusion.py        图谱确定性融合（实体/关系合并、稳定 ID、边规范化）
    keywords.py      双层关键词抽取
    index.py         索引编排
    modes.py         五模式路由
    graph.py         图一跳扩展
    fusion_retrieval.py  RRF/round-robin/rerank/预算截断
    retrieve.py      检索编排
  rag.py             MiniRAG 门面
  api/server.py      FastAPI: /documents /retrieve /health
```

## 存储方案

| 存储 | 后端 | 中间件 |
|---|---|---|
| 向量 | Milvus 三 Collection | Milvus |
| 图 | PG 普通表（graph_nodes/edges，边 src≤tgt 规范化，中文安全） | PostgreSQL |
| 元数据/分块 | PG documents/chunks | PostgreSQL |

图操作只需「一跳邻接 + 度数」，普通索引 SQL 足够，故图无需 Neo4j/AGE。

## 使用

### 配置

仓库只提交 `config.example.yaml` 模板；真实 `config.yaml` 保留在本地并已加入 `.gitignore`。

```bash
cd minirag
cp config.example.yaml config.yaml
# 编辑 config.yaml，或直接 export MINIRAG_* 环境变量
```

也可以通过 `MINIRAG_CONFIG=/path/to/config.yaml` 指定私有配置文件。

### 作为 Python 包

```python
from minirag import MiniRAG, DocumentInput, QueryParam

rag = MiniRAG()                 # 读取 config.yaml；不存在时回退到 config.example.yaml
await rag.startup()
await rag.index(DocumentInput(source="corpus/云网络告警处理手册.md"))
result = await rag.retrieve("BGP 中断如何处理", QueryParam(mode="mix", top_k=8))
# result.entities / result.relationships / result.chunks —— 只返回证据，不生成答案
await rag.shutdown()
```

### 作为 HTTP 服务

```bash
export MINIRAG_CHAT_API_KEY=...
export MINIRAG_EMBEDDING_API_KEY=...
export MINIRAG_RERANK_API_KEY=...
uvicorn minirag.api.server:app --port 8090
```

```bash
# 索引
curl -X POST localhost:8090/documents -d '{"source":"corpus/云网络告警处理手册.md"}'
# 检索（只返回证据）
curl -X POST localhost:8090/retrieve -d '{"query":"BGP 中断如何处理","mode":"mix","top_k":8}'
```

### 冒烟脚本

```bash
cd minirag
python scripts/smoke.py index ../corpus/云网络告警处理手册.md
python scripts/smoke.py retrieve "BGP 中断如何处理" --mode mix --top-k 8
```

## 依赖

Milvus 与 PostgreSQL 通过仓库根目录 `docker-compose.yml` 拉起。模型沿用现有
`config.example.yaml` 模板，真实 Key、endpoint 与 model id 经本地 `config.yaml`
或环境变量注入。
