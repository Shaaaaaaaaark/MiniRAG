#!/usr/bin/env python3
"""Validate retrieval benchmark evidence against normalized Feishu Markdown."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_blocks(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"<!-- feishu-block-id: ([^ ]+) -->\n", text)
    return {
        parts[index]: parts[index + 1].strip()
        for index in range(1, len(parts), 2)
        if parts[index + 1].strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("corpus/feishu/docs"),
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen_ids: set[str] = set()
    block_cache: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    for row in rows:
        case_id = str(row.get("id") or "")
        if not case_id or case_id in seen_ids:
            errors.append(f"invalid or duplicate id: {case_id!r}")
        seen_ids.add(case_id)

        evidence = row.get("gold_evidence") or []
        answerable = row.get("answerable")
        if answerable is True and not evidence:
            errors.append(f"{case_id}: answerable case has no gold evidence")
        if answerable is False and evidence:
            errors.append(f"{case_id}: unanswerable case has gold evidence")

        for item in evidence:
            token = str(item.get("doc_token") or "")
            block_id = str(item.get("block_id") or "")
            gold_text = str(item.get("gold_text") or "")
            doc_path = args.docs_dir / f"{token}.md"
            if not doc_path.exists():
                errors.append(f"{case_id}: missing document {token}")
                continue
            if token not in block_cache:
                block_cache[token] = load_blocks(doc_path)
            block_text = block_cache[token].get(block_id)
            if block_text is None:
                errors.append(f"{case_id}: missing block {token}#{block_id}")
            elif not gold_text or gold_text not in block_text:
                errors.append(
                    f"{case_id}: gold_text is not an exact substring of "
                    f"{token}#{block_id}"
                )

    if errors:
        for error in errors:
            print(error)
        return 1

    categories = Counter(str(row.get("category")) for row in rows)
    print(f"valid cases={len(rows)} categories={dict(categories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
