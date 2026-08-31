# Retrieval Benchmark

Business evaluation data is stored under `benchmarks/private/` and is ignored by Git.

Each JSONL row represents one retrieval test:

```json
{
  "id": "case-001",
  "question": "Example business question",
  "answerable": true,
  "category": "exact_api",
  "difficulty": "easy",
  "gold_evidence": [
    {
      "doc_token": "document token",
      "doc_title": "document title",
      "revision": "document revision",
      "block_id": "source block id",
      "source_url": "source URL with block anchor",
      "heading_path": "section path",
      "gold_text": "exact canonical evidence text"
    }
  ]
}
```

Recommended categories:

- `exact_api`: API names, fields, limits, and enum values.
- `error_code`: exact error-code lookup.
- `semantic_design`: design intent expressed with paraphrased questions.
- `multi_evidence`: questions requiring more than one evidence block.
- `unanswerable`: no supporting evidence in the indexed corpus.

## Evaluation layers

### 1. Deterministic retrieval evaluation

These metrics are the release gate. They use `block_id` and exact `gold_text`
and do not call an LLM:

| Metric | Purpose |
|---|---|
| `Hit@K` | Whether Top-K contains at least one gold evidence block |
| `MRR` | Rank of the first gold evidence block |
| `CharRecall@K` | Fraction of gold evidence characters covered by Top-K |
| `CharPrecision@K` | Fraction of retrieved characters supported by gold evidence |
| `UnanswerableFPR` | False-positive retrieval rate for unanswerable cases |
| `P50/P95Latency` | Retrieval latency distribution |

### 2. Ragas semantic evaluation

[Ragas](https://github.com/explodinggradients/ragas) is an optional evaluation
dependency, not a MiniRAG runtime dependency. For retrieval-only experiments,
map the dataset and API response as follows:

```text
question                                -> user_input
chunks[].text                           -> retrieved_contexts
concatenated gold_evidence[].gold_text  -> reference
chunks[].block_id                       -> Hit@K / MRR
```

The runner uses Ragas LLM-based context precision/recall as semantic
diagnostics for paraphrased or partially overlapping evidence. Exact block IDs
remain the deterministic source for Hit@K and MRR.

Ragas scores are not release gates until the judge model and prompts have been
calibrated against human-reviewed cases. Pin the judge model, prompt, metric
version, and evaluation dataset revision; cache judge responses for repeatable
comparisons. Private corpus content must only be sent to an approved internal
judge model.

Install the isolated evaluation dependencies with `uv`:

```bash
cd minirag
uv sync --extra eval
```

Run a small smoke experiment from the repository root:

```bash
uv run --project minirag --extra eval \
  python benchmarks/run_ragas.py --limit 3 --top-k 5
```

Run the complete private dataset:

```bash
uv run --project minirag --extra eval \
  python benchmarks/run_ragas.py \
  benchmarks/private/cloudwan_retrieval.jsonl \
  --base-url http://localhost:8090 \
  --top-k 5
```

The runner uses the private MiniRAG chat configuration as the Ragas judge,
disables Ragas usage tracking, and writes the full per-case JSON report under
`benchmarks/private/results/` by default. Use `--category`, `--limit`, and
`--judge-timeout` to control experiment scope and cost.

### 3. End-to-end answer evaluation

MiniRAG currently returns evidence and does not generate answers. After an
answer-generation service is connected, Ragas can additionally evaluate:

- faithfulness of the answer to retrieved evidence;
- answer relevancy to the user question;
- factual correctness against a reviewed reference answer.

Keep retrieval and answer metrics separate so a generation failure is not
misdiagnosed as a retrieval failure.

## Result policy

- Deterministic retrieval metrics are the primary regression and release gate.
- Ragas context metrics are secondary semantic diagnostics.
- LLM-judged metrics must always record judge model, prompt, and metric version.
- Experiment results should retain per-case evidence, timing, and effective
  retrieval configuration in addition to aggregate scores.

`run_ragas.py` currently reports `Hit@K`, MRR, unanswerable false-positive
rate, latency, and Ragas context precision/recall. Character-level metrics and
the HTML report remain planned benchmark components.

Validate a private dataset against the normalized corpus:

```bash
python3 benchmarks/validate_dataset.py \
  benchmarks/private/cloudwan_retrieval.jsonl
```
