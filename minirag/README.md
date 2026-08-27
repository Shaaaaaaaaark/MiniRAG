# MiniRAG — 轻量 RAG 检索服务

面向技术文档的完整 RAG 基线：支持飞书/Markdown 文档接入、Parent-Child 切片、
Dense + BM25 混合召回、Rerank、父块回填和引用溯源。系统固定使用
**Milvus + PostgreSQL**，对外提供结构化证据，不负责答案生成。

## 工程架构

```text
飞书/Markdown
  -> revision 增量同步
  -> 规范化 Block
  -> Parent-Child 切片
  -> Milvus Dense + BM25
  -> RRF -> Reranker -> PostgreSQL 父块回填
  -> 结构化证据与引用
```

固定数据模型：

| 模型 | 职责 |
|---|---|
| Document | 稳定 `source_id/doc_token`、revision、来源和处理状态 |
| Parent Chunk | 保留完整章节上下文，作为最终返回单位 |
| Child Chunk | 小粒度检索单位，只在 Milvus 建立 Dense/BM25 索引 |
| Evidence | 文本、分数、标题路径、revision 和可点击来源 |

生产设计原则：

1. 文档 ID 与 revision 分离，同一来源可幂等重建。
2. 索引失败回滚，服务启动时清理孤立图谱来源。
3. Reranker 不可用时返回融合排序，不阻断查询。
4. Embedding 模型或维度变化时必须重建向量索引。
5. 通过固定测试集评估 `Hit@K`、字符级 Precision/Recall、MRR 和 nDCG。

## 核心算法

默认 `text` 模式不调用 Chat LLM：

```text
query -> Embedding -> Dense + BM25 -> RRF -> Rerank
      -> Child 去重 -> Parent 回填 -> Token Budget
```

实体关系图是可选增强能力，通过 `graph_enabled=true` 开启。检索模式：

| 模式 | 逻辑 |
|---|---|
| `text`（默认） | Dense + BM25 + RRF + Rerank |
| `local` | low 关键词 → 实体向量 → 图取邻边(关系) + 实体来源 chunk |
| `global` | high 关键词 → 关系向量 → 取两端实体 + 关系来源 chunk |
| `hybrid` | local + global，round-robin 合并 |
| `mix` | hybrid + 文本混合召回 |
| `naive` | `text` 的兼容别名 |

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
    chunker.py       解析(MD/TXT) + 标题层级/Parent-Child 切分
    tokenizer.py     tiktoken 计数
    prompts.py       抽取 / 双层关键词 prompt
    extract.py       实体关系抽取（LLM + 一次 gleaning）
    fusion.py        图谱确定性融合（实体/关系合并、稳定 ID、边规范化）
    keywords.py      双层关键词抽取
    index.py         索引编排
    modes.py         文本检索 + 可选图谱模式路由
    graph.py         图一跳扩展
    fusion_retrieval.py  RRF/round-robin/rerank/预算截断
    retrieve.py      检索编排
  rag.py             MiniRAG 门面
  api/server.py      FastAPI: /documents /retrieve /health
  integrations/
    feishu.py        飞书目录/Docx XML 读取与规范化
    feishu_sync.py   revision 增量同步 + manifest
    feishu_jit.py    指定飞书文档的 JIT 检索
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
未提供 `config.yaml` 时，服务会直接报错，不会自动使用模板文件。

技术文档推荐使用 `chunker: parent_child`：子块参与检索，命中后返回包含完整上下文的父章节。

### 作为 Python 包

```python
from minirag import MiniRAG, DocumentInput, QueryParam

rag = MiniRAG()                 # 读取 config.yaml，或使用 MINIRAG_CONFIG 指定路径
await rag.startup()
await rag.index(DocumentInput(source="/path/to/document.md"))
result = await rag.retrieve("BGP 中断如何处理", QueryParam(mode="text", top_k=8))
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
curl -X POST localhost:8090/documents -d '{"source":"/path/to/document.md"}'
# 检索（只返回证据）
curl -X POST localhost:8090/retrieve -d '{"query":"BGP 中断如何处理","mode":"text","top_k":8}'
```

## 飞书文档

### 增量同步目录

同步依赖本机已安装并登录的 `lark-cli`。默认只生成规范 Markdown 和
`manifest.json`；加 `--index` 才会连接 MiniRAG 并索引变更文档。

```bash
cd minirag
python scripts/sync_feishu.py \
  --folder "https://example.feishu.cn/drive/folder/..." \
  --index
```

同步产物默认位于仓库 `corpus/feishu/`：

```text
corpus/feishu/
  manifest.json
  docs/<doc_token>.md
```

`manifest.json` 使用 `doc_token + revision_id + modified_time` 判断增量更新。
只有显式传入 `--prune` 才会删除已经移出目录的本地语料和索引。

### JIT 精确读取

在私有 `config.yaml` 中启用：

```yaml
feishu:
  enabled: true
  cli_path: lark-cli
  identity: user
  timeout_seconds: 120
```

显式调用：

```bash
curl -X POST localhost:8090/feishu/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"source_url":"https://example.feishu.cn/docx/xxx","query":"灰度流程是什么","top_k":5}'
```

`POST /retrieve` 也支持 `source_url`；查询文本中直接包含飞书 Docx/Wiki URL 时会自动走 JIT。
JIT 使用当前 `lark-cli` 身份实时读取最新 revision，再对子块召回并返回父章节。

> 当前服务没有文档级 ACL。离线索引只适合单用户或固定授权目录。多人服务必须增加
> 请求者权限过滤。Docker 镜像默认不包含 `lark-cli`，容器内 JIT 默认关闭。

### 冒烟脚本

```bash
cd minirag
python scripts/smoke.py index /path/to/document.md
python scripts/smoke.py retrieve "BGP 中断如何处理" --mode text --top-k 8
```

## 依赖

Milvus 与 PostgreSQL 通过仓库根目录 `docker-compose.yml` 拉起。模型沿用现有
`config.example.yaml` 模板，真实 Key、endpoint 与 model id 经本地 `config.yaml`
或环境变量注入。
