#!/usr/bin/env python3
"""Build a balanced mixed_ua suite from existing sample pools.

Policy (default --per-bucket 32):
  chat / instruct  — hand-written only (base + extras)
  knowledge        — custom + ZNO (cap)
  code             — hand-written exec-tested only (cap); UA-Code kept aside
  translate        — prefer custom, then FLORES (cap)
  alignment        — UAlign, balanced ethics/social if possible (cap)

Usage:
  python scripts/build_balanced_suite.py --per-bucket 32 \\
    --out benchmarks/mixed_ua_v4_balanced.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ("chat", "translate", "instruct", "knowledge", "code", "alignment")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def source_rank(row: dict) -> int:
    """Lower = preferred when capping."""
    rid = str(row.get("id", ""))
    notes = str(row.get("notes", ""))
    if rid.startswith(("chat-", "tr-", "if-", "know-", "code-")) and "flores" not in rid:
        if rid.startswith("code-") and rid[5:].isdigit() and int(rid[5:]) <= 30:
            return 0  # exec-tested hand-written
        if not rid.startswith("code-"):
            return 0
    if "hand" in notes or "custom" in notes:
        return 0
    if rid.startswith("zno-"):
        return 1
    if rid.startswith("flores-"):
        return 2
    if rid.startswith("ualign-"):
        return 2
    if rid.startswith("uacode-"):
        return 9  # excluded from scored balanced suite by default
    return 5


def pick(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    # stable: sort by rank, then id; within same rank shuffle with seed
    by_rank: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_rank[source_rank(r)].append(r)
    chosen: list[dict] = []
    for rank in sorted(by_rank):
        chunk = by_rank[rank][:]
        chunk.sort(key=lambda r: str(r.get("id")))
        rng.shuffle(chunk)
        for r in chunk:
            if len(chosen) >= n:
                break
            chosen.append(r)
        if len(chosen) >= n:
            break
    return chosen[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-bucket", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-uacode", action="store_true", help="Allow UA-Code into code bucket")
    p.add_argument("--out", type=Path, default=Path("benchmarks/mixed_ua_v4_balanced.jsonl"))
    args = p.parse_args()

    # Prefer FULL corpora (thousands) when present; fall back to tiny samples/.
    corpus = ROOT / "benchmarks" / "corpus"
    paths = [
        ROOT / "benchmarks/mixed_ua_v0.jsonl",
        ROOT / "benchmarks/samples/chat_extra_v1.jsonl",
        ROOT / "benchmarks/samples/instruct_extra_v1.jsonl",
        ROOT / "benchmarks/samples/code_extra_v1.jsonl",
        ROOT / "benchmarks/samples/code_extra_v2.jsonl",
    ]
    # knowledge / translate / alignment / code warehouses
    for name in (
        "zno_knowledge_full.jsonl",
        "flores_translate_full.jsonl",
        "ualign_ethics_full.jsonl",
        "ualign_social_full.jsonl",
    ):
        paths.append(corpus / name)
    # tiny samples only if corpus missing (legacy)
    if not (corpus / "zno_knowledge_full.jsonl").exists():
        paths.append(ROOT / "benchmarks/samples/zno_knowledge_v1.jsonl")
    if not (corpus / "flores_translate_full.jsonl").exists():
        paths.append(ROOT / "benchmarks/samples/flores_translate_v1.jsonl")
    if not (corpus / "ualign_ethics_full.jsonl").exists():
        paths.append(ROOT / "benchmarks/samples/ualign_v1.jsonl")
    if args.include_uacode:
        uacode_full = corpus / "uacode_problems_full.jsonl"
        paths.append(uacode_full if uacode_full.exists() else ROOT / "benchmarks/samples/uacode_easy_v1.jsonl")

    pool: dict[str, dict] = {}
    for path in paths:
        for row in load_jsonl(path):
            rid = str(row.get("id"))
            if rid and rid not in pool:
                # drop UA-Code from scored suite unless explicitly included
                if str(row.get("id", "")).startswith("uacode-") and not args.include_uacode:
                    continue
                pool[rid] = row

    by_b: dict[str, list[dict]] = defaultdict(list)
    for row in pool.values():
        b = row.get("bucket")
        if b in BUCKETS:
            by_b[b].append(row)

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    report: dict[str, dict] = {}
    for b in BUCKETS:
        selected = pick(by_b[b], args.per_bucket, rng)
        # keep deterministic order within bucket by id
        selected.sort(key=lambda r: str(r.get("id")))
        out_rows.extend(selected)
        report[b] = {
            "selected": len(selected),
            "pool": len(by_b[b]),
            "shortfall": max(0, args.per_bucket - len(selected)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f"# mixed_ua balanced — {args.per_bucket}/bucket, seed={args.seed}; do not hand-edit\n"
        )
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "n": len(out_rows),
                "per_bucket_target": args.per_bucket,
                "by_bucket": report,
                "include_uacode": args.include_uacode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
