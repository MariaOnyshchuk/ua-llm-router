#!/usr/bin/env python3
"""Draw a screening suite (≈1000–1500 items/bucket) from the corpus warehouse.

Small buckets (chat, hand-written code, IFEval) contribute the whole pool.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ("chat", "translate", "instruct", "knowledge", "code", "alignment")

CORPUS_FILES = (
    "zno_knowledge_full.jsonl",
    "flores_translate_full.jsonl",
    "ualign_ethics_full.jsonl",
    "ualign_social_full.jsonl",
    "uacode_problems_full.jsonl",
    "ifeval_ukr_instruct.jsonl",
    "belebele_uk_knowledge.jsonl",
    "mmlu_ukr_knowledge.jsonl",
    "arc_challenge_ukr_knowledge.jsonl",
    "wmt22_translate.jsonl",
)

HAND_FILES = (
    ROOT / "benchmarks/mixed_ua_v0.jsonl",
    ROOT / "benchmarks/samples/chat_extra_v1.jsonl",
    ROOT / "benchmarks/samples/instruct_extra_v1.jsonl",
    ROOT / "benchmarks/samples/code_extra_v1.jsonl",
    ROOT / "benchmarks/samples/code_extra_v2.jsonl",
)

SOURCE_RE = re.compile(r"source=([^\s|]+)")


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


def source_key(row: dict) -> str:
    notes = str(row.get("notes") or "")
    m = SOURCE_RE.search(notes)
    if m:
        return m.group(1)
    rid = str(row.get("id") or "")
    return rid.split("-")[0] if rid else "unknown"


def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n >= len(rows):
        return sorted(rows, key=lambda r: str(r.get("id")))
    by_src: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_src[source_key(row)].append(row)
    keys = sorted(by_src)
    per = max(1, n // max(1, len(keys)))
    chosen: list[dict] = []
    leftover: list[dict] = []
    for key in keys:
        chunk = by_src[key][:]
        chunk.sort(key=lambda r: str(r.get("id")))
        rng.shuffle(chunk)
        take = min(len(chunk), per)
        chosen.extend(chunk[:take])
        leftover.extend(chunk[take:])
    rng.shuffle(leftover)
    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
    elif len(chosen) < n:
        chosen.extend(leftover[: n - len(chosen)])
    chosen.sort(key=lambda r: str(r.get("id")))
    return chosen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-bucket", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-uacode", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "mixed_ua_v6_screen.jsonl",
    )
    args = p.parse_args()

    corpus = ROOT / "benchmarks" / "corpus"
    pool: dict[str, dict] = {}
    for path in HAND_FILES:
        for row in load_jsonl(path):
            rid = str(row.get("id") or "")
            if rid:
                pool.setdefault(rid, row)
    for name in CORPUS_FILES:
        for row in load_jsonl(corpus / name):
            rid = str(row.get("id") or "")
            if not rid:
                continue
            if rid.startswith("uacode-") and not args.include_uacode:
                continue
            pool.setdefault(rid, row)

    by_b: dict[str, list[dict]] = defaultdict(list)
    for row in pool.values():
        b = row.get("bucket")
        if b in BUCKETS:
            by_b[b].append(row)

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    report: dict[str, dict] = {}
    for bucket in BUCKETS:
        selected = stratified_sample(by_b[bucket], args.per_bucket, rng)
        out_rows.extend(selected)
        report[bucket] = {
            "selected": len(selected),
            "pool": len(by_b[bucket]),
            "used_all": len(by_b[bucket]) <= args.per_bucket,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f"# mixed_ua_v6_screen — up to {args.per_bucket}/bucket, seed={args.seed}\n"
        )
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"wrote": str(args.out), "n": len(out_rows), "by_bucket": report},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
