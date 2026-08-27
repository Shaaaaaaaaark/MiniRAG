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

The primary retrieval metrics should use `block_id` and exact `gold_text`; an LLM judge is optional.

Validate a private dataset against the normalized corpus:

```bash
python3 benchmarks/validate_dataset.py \
  benchmarks/private/cloudwan_retrieval.jsonl
```
