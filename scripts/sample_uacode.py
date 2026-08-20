#!/usr/bin/env python3
"""Sample easy UA-Code-Bench problems into our code JSONL schema.

Dataset card / license:
  https://huggingface.co/datasets/NLPForUA/ua-code-bench
  CC-BY-NC-4.0; problem statements © Eolymp (often URL-referenced).

Schema (as of 2026): problem_url, title, statement_summary, complexity (1–5),
model runs… We dedupe by problem_url and keep complexity bands 1–2.

IMPORTANT: rows only include statement_summary, not full Eolymp text / hidden tests.
They are imported for inventory / future judge wiring — do NOT treat as
exec-scored quality until Eolymp judge is connected. Hand-written code-* items
remain the scored code bucket.

Usage:
  export HF_TOKEN=hf_...
  python scripts/sample_uacode.py --n 32 --bands 1,2 \\
    --out benchmarks/samples/uacode_easy_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


def _pid_from_url(url: str) -> str:
    m = re.search(r"/problems/(\d+)", url or "")
    return m.group(1) if m else re.sub(r"\W+", "_", url)[:40]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="NLPForUA/ua-code-bench")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--bands", default="1,2", help="complexity ints (1=easiest)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("benchmarks/samples/uacode_easy_v1.jsonl"))
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets", file=sys.stderr)
        sys.exit(2)

    bands = {int(x) for x in args.bands.split(",") if x.strip()}
    ds = load_dataset(args.dataset, split="train", token=os.getenv("HF_TOKEN"))

    seen: set[str] = set()
    pool: list[dict] = []
    for row in ds:
        r = dict(row)
        try:
            diff = int(r.get("complexity"))
        except (TypeError, ValueError):
            continue
        if diff not in bands:
            continue
        if r.get("media_needed"):
            continue
        url = str(r.get("problem_url") or "")
        title = str(r.get("title") or "").strip()
        summary = str(r.get("statement_summary") or "").strip()
        if not url or not summary or len(summary) < 40:
            continue
        pid = _pid_from_url(url)
        if pid in seen:
            continue
        seen.add(pid)
        pool.append(
            {
                "id": f"uacode-{pid}",
                "bucket": "code",
                "prompt": (
                    "Розв'яжи задачу змагального програмування. "
                    "Напиши повне рішення на Python 3 (stdin → stdout). "
                    "Лише код у блоці ```python.\n\n"
                    f"Назва: {title}\n"
                    f"Умова (короткий виклад з UA-Code-Bench): {summary}\n"
                    f"Повне джерело: {url}"
                ),
                "reference": "",
                "notes": (
                    f"source=ua-code-bench complexity={diff} title={title} "
                    f"url={url} score_mode=deferred_eolymp"
                ),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected = pool[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "n": len(selected),
                "unique_problems_in_bands": len(pool),
                "bands": sorted(bands),
                "dataset": args.dataset,
                "warning": (
                    "UA-Code rows are summary+URL only; scoring deferred to Eolymp. "
                    "Keep hand-written code-* as the scored code bucket."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
