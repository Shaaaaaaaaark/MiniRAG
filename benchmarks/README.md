# 检索评测

业务评测数据保存在 `benchmarks/private/`，该目录已被 Git 忽略，不会进入公开仓库。

每行 JSONL 表示一个检索测试用例：

```json
{
  "id": "case-001",
  "question": "示例业务问题",
  "answerable": true,
  "category": "exact_api",
  "difficulty": "easy",
  "gold_evidence": [
    {
      "doc_token": "文档 token",
      "doc_title": "文档标题",
      "revision": "文档 revision",
      "block_id": "原文 block ID",
      "source_url": "带 block 锚点的原文链接",
      "heading_path": "章节路径",
      "gold_text": "规范语料中的证据原文"
    }
  ]
}
```

测试类别：

- `exact_api`：API 名称、字段、限制和枚举值。
- `error_code`：错误码精确查询。
- `semantic_design`：使用改写问题检索设计意图。
- `multi_evidence`：需要多段证据的问题。
- `unanswerable`：索引语料中没有答案的问题。

## 评测分层

### 1. 确定性检索评测

这些指标是主要回归和发布门禁，使用 `block_id` 和精确 `gold_text` 计算，
不调用 LLM：

| 指标 | 作用 |
|---|---|
| `Hit@K` | Top-K 是否包含至少一个正确证据块 |
| `MRR` | 第一条正确证据的排名 |
| `CharRecall@K` | Top-K 覆盖了多少 gold 原文 |
| `CharPrecision@K` | 召回文本中有多少属于 gold 原文 |
| `UnanswerableFPR` | 无答案问题被误召回的比例 |
| `P50/P95Latency` | 检索延迟分布 |

### 2. Ragas 语义评测

[Ragas](https://github.com/explodinggradients/ragas) 是可选评测依赖，不属于
MiniRAG 生产运行依赖。检索评测的数据映射如下：

```text
question                                -> user_input
chunks[].text                           -> retrieved_contexts
拼接后的 gold_evidence[].gold_text      -> reference
chunks[].block_id                       -> Hit@K / MRR
```

Runner 使用 Ragas 的 LLM Context Precision 和 Context Recall，诊断改写问题、
语义等价证据或部分重叠证据。精确的 `block_id` 仍然是 Hit@K 和 MRR 的确定性依据。

在 Judge 模型与 Prompt 经过人工样本校准前，Ragas 分数不作为发布门禁。实验必须
记录 Judge 模型、Prompt、指标版本和数据集 revision。私有语料只能发送给经过批准
的内部 Judge 模型。

使用 `uv` 安装隔离的评测依赖：

```bash
cd minirag
uv sync --extra eval
```

从仓库根目录运行 3 条冒烟评测：

```bash
uv run --project minirag --extra eval \
  python benchmarks/run_ragas.py --limit 3 --top-k 5
```

运行完整私有数据集：

```bash
uv run --project minirag --extra eval \
  python benchmarks/run_ragas.py \
  benchmarks/private/cloudwan_retrieval.jsonl \
  --base-url http://localhost:8090 \
  --top-k 5
```

Runner 使用 MiniRAG 私有 Chat 配置作为 Ragas Judge，默认关闭 Ragas 使用情况
上报，并将匿名本地标识放入系统临时目录。逐题 JSON 报告默认写入
`benchmarks/private/results/`。可使用 `--category`、`--limit` 和
`--judge-timeout` 控制范围、成本与超时。

### 3. 端到端问答评测

MiniRAG 当前只返回证据，不生成答案。接入答案生成服务后，可以继续使用 Ragas
评估：

- 答案是否忠实于召回证据；
- 答案是否回应用户问题；
- 答案与人工参考答案是否事实一致。

检索指标和答案指标必须分开统计，避免把生成错误误判为检索错误。

## 结果约定

- 确定性检索指标是主要回归和发布门禁。
- Ragas Context 指标作为语义诊断。
- LLM Judge 结果必须记录模型、Prompt 和指标版本。
- 除聚合分数外，还应保留逐题证据、耗时和实际检索配置。

当前 `run_ragas.py` 已输出 `Hit@K`、MRR、无答案误召回率、延迟，以及 Ragas
Context Precision/Recall。字符级指标和 HTML 报告仍待实现。

## 数据集校验

校验私有测试集中的 gold evidence 是否仍能在规范语料中定位：

```bash
python3 benchmarks/validate_dataset.py \
  benchmarks/private/cloudwan_retrieval.jsonl
```
