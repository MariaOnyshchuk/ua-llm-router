#!/usr/bin/env python3
"""Paired comparison of two systems on one split of mixed_ua_v7.

For each claim bucket and for the macro of the four: mean of A, mean of B, the
difference and a paired bootstrap 95% interval. A verdict is "A higher" or
"A lower" only when the interval excludes 0; otherwise "not distinguishable".

Reads item-level score files (ids and scores, no prompts). Rows are filtered by
the split recorded in the benchmark file before anything is computed, so a
whole-pool score file can be used without touching the other split.

  python scripts/analysis/v7_compare.py \\
      --a results/v7_dev_router/scores_router_dev_detail.jsonl \\
      --b results/v7_dev_router/scores_mamay12_dev_detail.jsonl \\
      --b-override results/v7_dev_router/scores_mamay12_fewshot_dev_detail.jsonl

The split defaults to dev. Use --split test once, after the router is frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLAIM_BUCKETS = ("knowledge", "translate", "alignment", "instruct")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def load_scores(path: Path, system: str | None, keep: set[str]) -> tuple[str, dict[str, float]]:
    rows = [r for r in read_jsonl(path) if "score" in r and r["id"] in keep]
    systems = sorted({r["system"] for r in rows})
    if system is None:
        if len(systems) != 1:
            raise SystemExit(f"{path} holds systems {systems}; choose one with --a-system / --b-system")
        system = systems[0]
    scores = {r["id"]: float(r["score"]) for r in rows if r["system"] == system}
    if not scores:
        raise SystemExit(f"no rows for system {system!r} in {path}")
    return system, scores


def interval(values: list[float], rng: random.Random, n: int) -> list[float]:
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)]


def verdict(ci: list[float]) -> str:
    return "A higher" if ci[0] > 0 else "A lower" if ci[1] < 0 else "not distinguishable"


def compare(a: dict[str, float], b: dict[str, float], bucket_of: dict[str, str], seed: int,
            n_boot: int) -> dict[str, Any]:
    rng = random.Random(seed)
    ids = {bk: sorted(i for i in a if i in b and bucket_of[i] == bk) for bk in CLAIM_BUCKETS}
    rows: dict[str, Any] = {}
    for bucket, members in ids.items():
        if not members:
            raise SystemExit(f"no shared items in bucket {bucket}")
        diffs = [a[i] - b[i] for i in members]
        ci = interval(diffs, rng, n_boot) if any(diffs) else [0.0, 0.0]
        rows[bucket] = {"n": len(members), "a": round(st.mean(a[i] for i in members), 4),
                        "b": round(st.mean(b[i] for i in members), 4), "diff": round(st.mean(diffs), 4),
                        "ci95": ci, "verdict": verdict(ci)}
    # Macro: resample inside each bucket, then average the four bucket differences.
    draws = sorted(
        st.mean(st.mean(a[i] - b[i] for i in rng.choices(members, k=len(members))) for members in ids.values())
        for _ in range(n_boot)
    )
    ci = [round(draws[int(0.025 * n_boot)], 4), round(draws[int(0.975 * n_boot)], 4)]
    rows["macro4"] = {"n": sum(len(m) for m in ids.values()),
                      "a": round(st.mean(rows[bk]["a"] for bk in CLAIM_BUCKETS), 4),
                      "b": round(st.mean(rows[bk]["b"] for bk in CLAIM_BUCKETS), 4),
                      "diff": round(st.mean(rows[bk]["diff"] for bk in CLAIM_BUCKETS), 4),
                      "ci95": ci, "verdict": verdict(ci)}
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default="benchmarks/mixed_ua_v7.jsonl")
    p.add_argument("--split", default="dev", choices=("dev", "test"))
    p.add_argument("--a", required=True, help="item-level scores of system A")
    p.add_argument("--b", required=True, help="item-level scores of system B")
    p.add_argument("--a-system", default=None)
    p.add_argument("--b-system", default=None)
    p.add_argument("--b-override", default=None,
                   help="scores that replace B's rows for the ids they hold (for example B with the "
                        "router's alignment prompt)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    bench = read_jsonl(ROOT / args.bench)
    bucket_of = {r["id"]: r["bucket"] for r in bench if r.get("split") == args.split}
    keep = set(bucket_of)
    name_a, a = load_scores(ROOT / args.a, args.a_system, keep)
    name_b, b = load_scores(ROOT / args.b, args.b_system, keep)
    overridden = 0
    if args.b_override:
        _, extra = load_scores(ROOT / args.b_override, args.b_system or name_b, keep)
        overridden = len(extra)
        b = {**b, **extra}
    rows = compare(a, b, bucket_of, args.seed, args.n_boot)
    result = {"suite": Path(args.bench).stem, "split": args.split, "a": name_a, "b": name_b,
              "a_file": args.a, "b_file": args.b, "b_override": args.b_override,
              "b_rows_overridden": overridden, "seed": args.seed, "n_boot": args.n_boot, "rows": rows}

    print(f"\nA = {name_a}, B = {name_b}, mixed_ua_v7 {args.split}\n")
    print("| bucket | n | A | B | A − B | 95% interval | verdict |")
    print("|---|---:|---:|---:|---:|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['n']} | {r['a']:.3f} | {r['b']:.3f} | {r['diff']:+.3f} | "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['verdict']} |")
    if args.out:
        out = ROOT / args.out
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
