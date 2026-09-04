#!/usr/bin/env python3
"""Import English coding benchmarks (HumanEval) as scored JSONL.

Not a web scrape: official Hugging Face `openai/openai_humaneval` (MIT).
UA-Code stays unscored (no Eolymp judge). These rows have unit tests.

Usage:
  python scripts/import_english_code.py
  python scripts/import_english_code.py --append-to-v4 32 --out-suite benchmarks/mixed_ua_v5.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


PROMPT_WRAP = (
    "Write a complete Python 3 solution. Return only a ```python code block.\n"
    "Do not explain.\n\n"
    "{prompt}"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-corpus", type=Path, default=ROOT / "benchmarks/corpus/humaneval_code_en.jsonl")
    p.add_argument("--append-to-v4", type=int, default=32, help="How many HumanEval items to add on top of v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-suite", type=Path, default=ROOT / "benchmarks/mixed_ua_v5.jsonl")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets", file=sys.stderr)
        sys.exit(2)

    ds = load_dataset("openai/openai_humaneval", split="test", token=os.getenv("HF_TOKEN"))
    rows: list[dict] = []
    for ex in ds:
        tid = str(ex.get("task_id") or "")
        num = tid.split("/")[-1] if "/" in tid else tid
        prompt = str(ex.get("prompt") or "")
        test = str(ex.get("test") or "")
        entry = str(ex.get("entry_point") or "")
        if not prompt or not test or not entry:
            continue
        rows.append(
            {
                "id": f"humaneval-{int(num):03d}" if num.isdigit() else f"humaneval-{num}",
                "bucket": "code",
                "prompt": PROMPT_WRAP.format(prompt=prompt.rstrip() + "\n"),
                "reference": "",
                "he_prompt": prompt,
                "he_test": test,
                "he_entry_point": entry,
                "notes": f"source=openai_humaneval task_id={tid} score_mode=humaneval lang=en",
            }
        )

    args.out_corpus.parent.mkdir(parents=True, exist_ok=True)
    with args.out_corpus.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    v4 = ROOT / "benchmarks/mixed_ua_v4_balanced.jsonl"
    v4_rows = []
    for line in v4.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            v4_rows.append(json.loads(line))

    rng = random.Random(args.seed)
    extra = rows[:]
    rng.shuffle(extra)
    extra = extra[: max(0, args.append_to_v4)]
    extra.sort(key=lambda r: str(r["id"]))

    args.out_suite.parent.mkdir(parents=True, exist_ok=True)
    with args.out_suite.open("w", encoding="utf-8") as f:
        f.write(
            f"# mixed_ua_v5 = v4 192 + {len(extra)} English HumanEval (seed={args.seed}); "
            "chat/instruct still 32 (hand-written cap)\n"
        )
        for row in v4_rows + extra:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "corpus": str(args.out_corpus),
                "corpus_n": len(rows),
                "suite": str(args.out_suite),
                "suite_n": len(v4_rows) + len(extra),
                "v4_n": len(v4_rows),
                "humaneval_added": len(extra),
                "ids_added": [r["id"] for r in extra],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
