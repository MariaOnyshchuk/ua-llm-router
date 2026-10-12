#!/usr/bin/env python3
"""Build mixed_ua_v7: the whole screening pool, split into dev and test.

v6 was cut from the same pool with a selection step that looked at model
scores, and that step was inverted (it kept the items where Mamay-4B, Lapa and
Aya score the same). v7 uses no model score at all: every pool item is kept
and assigned to dev or test at random inside its source stratum.

Items that also appear in mixed_ua_v4_balanced go to dev, because routing
rules, prompts and few-shot examples were tuned on v4.

Items that share text stay in one split: the two directions of one FLORES or
WMT-22 sentence pair, and items with an identical prompt.

    python scripts/benchmarks/build_mixed_ua_v7.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "benchmarks" / "mixed_ua_v6_screen.jsonl"
TUNED_ON = ROOT / "benchmarks" / "mixed_ua_v4_balanced.jsonl"
OUT = ROOT / "benchmarks" / "mixed_ua_v7.jsonl"
CLAIM_BUCKETS = ("knowledge", "translate", "alignment", "instruct")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def stratum(row: dict[str, Any]) -> str:
    """Bucket plus source; UAlign also keeps its subset (social, ethics)."""
    parts = str(row["id"]).split("-")
    source = "-".join(parts[:2]) if parts[0] == "ualign" else parts[0]
    return f"{row['bucket']}/{source}"


def leak_keys(row: dict[str, Any]) -> list[str]:
    """Keys shared by items that must not be separated by the split."""
    keys = ["prompt:" + re.sub(r"\s+", " ", str(row["prompt"])).strip().lower()]
    match = re.fullmatch(r"(flores|wmt22)-(?:en-uk|uk-en)-(\w+)-(\d+)", str(row["id"]))
    if match:
        keys.append("pair:" + "-".join(match.groups()))
    return keys


def leak_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Union items that share any key (union-find), keyed by the smallest member id."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in rows:
        for key in leak_keys(row):
            parent[find(key)] = find("id:" + str(row["id"]))
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        members[find("id:" + str(row["id"]))].append(row)
    return {min(str(r["id"]) for r in group): group for group in members.values()}


def assign_splits(rows: list[dict[str, Any]], forced_dev: set[str], dev_share: float,
                  seed: int) -> dict[str, str]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[stratum(row)].append(row)
    split: dict[str, str] = {}
    for name in sorted(by_stratum):
        groups = leak_groups(by_stratum[name])
        keys = sorted(groups)
        random.Random(f"{seed}:{name}").shuffle(keys)
        # Groups that hold a tuned-on item go to dev first; then fill dev up to its share.
        keys.sort(key=lambda k: not any(r["id"] in forced_dev for r in groups[k]))
        target = round(len(by_stratum[name]) * dev_share)
        n_dev = 0
        for key in keys:
            forced = any(r["id"] in forced_dev for r in groups[key])
            to_dev = forced or n_dev < target
            for row in groups[key]:
                split[row["id"]] = "dev" if to_dev else "test"
            n_dev += len(groups[key]) if to_dev else 0
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=str(POOL))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--dev-share", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.pool))
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids in the pool")
    forced_dev = {r["id"] for r in read_jsonl(TUNED_ON)} & set(ids)
    split = assign_splits(rows, forced_dev, args.dev_share, args.seed)

    out = Path(args.out)
    rows = sorted(rows, key=lambda r: (r["bucket"], str(r["id"])))
    with out.open("w", encoding="utf-8") as handle:
        handle.write(f"# mixed_ua_v7 — whole screening pool, no score-based selection; "
                     f"seed={args.seed} dev_share={args.dev_share:.4f}; do not hand-edit\n")
        for row in rows:
            handle.write(json.dumps({**row, "split": split[row["id"]]}, ensure_ascii=False) + "\n")

    counts: dict[str, Counter] = {s: Counter() for s in ("dev", "test")}
    strata: dict[str, Counter] = {s: Counter() for s in ("dev", "test")}
    for row in rows:
        counts[split[row["id"]]][row["bucket"]] += 1
        strata[split[row["id"]]][stratum(row)] += 1
    meta = {
        "suite": "mixed_ua_v7",
        "pool": str(Path(args.pool).relative_to(ROOT)),
        "pool_sha1": hashlib.sha1(Path(args.pool).read_bytes()).hexdigest(),
        "selection": "none (every pool item kept); split is random inside bucket/source strata",
        "kept_together": "both directions of a FLORES or WMT-22 sentence pair; items with an identical prompt",
        "seed": args.seed,
        "dev_share": args.dev_share,
        "n": len(rows),
        "forced_dev_from_v4": len(forced_dev),
        "claim_buckets": list(CLAIM_BUCKETS),
        "appendix_buckets": ["chat", "code"],
        "by_split_bucket": {s: dict(sorted(c.items())) for s, c in counts.items()},
        "by_split_stratum": {s: dict(sorted(c.items())) for s, c in strata.items()},
    }
    meta_path = out.with_name(out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for s in ("dev", "test"):
        print(s, sum(counts[s].values()), dict(sorted(counts[s].items())))
    print(f"wrote {out} and {meta_path}")


if __name__ == "__main__":
    main()
