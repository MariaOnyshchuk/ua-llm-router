#!/usr/bin/env python3
"""Paired item-level bootstrap for system comparisons on mixed_ua_v4_balanced.

Reads per-item `scores_detail.jsonl` files (fields: id, bucket, score), averages
repeats per item, then resamples items with replacement (stratified by bucket,
so every bucket keeps its 32 items). Reports the mean difference A - B with a
95% percentile CI and the bootstrap share of resamples where A <= B.

Usage:
  python scripts/bootstrap_compare.py            # default comparisons
  python scripts/bootstrap_compare.py --n-boot 20000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = "results"

# label -> glob(s) of per-item scored files.
SYSTEMS = {
    "mamay12 (S2, 3 reps)": [f"{R}/week_9_baselines/scored/*scores_detail.jsonl"],
    "router rules v2 + few-shot, FP8 one card (3 reps)": [
        f"{R}/week_11_pack_fp8/scored/*scores_detail.jsonl"
    ],
    "router rules v1, pre-fix (1 rep)": [f"{R}/week_5_router_v4/scored/*scores_detail.jsonl"],
    "router ensemble vote (1 rep)": [f"{R}/week_7_ensemble/scores_detail.jsonl"],
}

# (A, B) pairs; difference is A - B.
PAIRS = [
    ("router rules v2 + few-shot, FP8 one card (3 reps)", "mamay12 (S2, 3 reps)"),
    ("router ensemble vote (1 rep)", "mamay12 (S2, 3 reps)"),
    ("router rules v1, pre-fix (1 rep)", "mamay12 (S2, 3 reps)"),
    ("router ensemble vote (1 rep)", "router rules v2 + few-shot, FP8 one card (3 reps)"),
]


def load_items(patterns: list[str]) -> dict[str, tuple[str, float]]:
    """id -> (bucket, mean score over repeats)."""
    files = sorted({p for pat in patterns for p in glob(str(ROOT / pat))})
    if not files:
        raise FileNotFoundError(f"no files for {patterns}")
    acc: dict[str, list[float]] = defaultdict(list)
    bucket: dict[str, str] = {}
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            acc[row["id"]].append(float(row["score"]))
            bucket[row["id"]] = row["bucket"]
    return {i: (bucket[i], float(np.mean(v))) for i, v in acc.items()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, groups: list[np.ndarray], n_boot: int, rng):
    """Stratified paired bootstrap of mean(a) - mean(b)."""
    diffs = np.empty(n_boot)
    d = a - b
    for k in range(n_boot):
        parts = [d[rng.choice(g, size=len(g), replace=True)] for g in groups]
        diffs[k] = np.concatenate(parts).mean()
    return diffs


def summarize(name_a, name_b, A, B, n_boot, seed):
    ids = sorted(set(A) & set(B))
    if len(ids) < 10:
        raise ValueError(f"too few shared items for {name_a} vs {name_b}: {len(ids)}")
    rng = np.random.default_rng(seed)
    a = np.array([A[i][1] for i in ids])
    b = np.array([B[i][1] for i in ids])
    buckets = np.array([A[i][0] for i in ids])
    rows = []
    for scope in ["overall"] + sorted(set(buckets)):
        idx = np.arange(len(ids)) if scope == "overall" else np.where(buckets == scope)[0]
        if scope == "overall":
            groups = [np.where(buckets == bk)[0] for bk in sorted(set(buckets))]
        else:
            groups = [idx]
        diffs = paired_bootstrap(a, b, groups, n_boot, rng)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append(
            {
                "A": name_a,
                "B": name_b,
                "scope": scope,
                "n_items": int(len(idx)),
                "mean_A": round(float(a[idx].mean()), 4),
                "mean_B": round(float(b[idx].mean()), 4),
                "diff_A_minus_B": round(float(a[idx].mean() - b[idx].mean()), 4),
                "ci95_low": round(float(lo), 4),
                "ci95_high": round(float(hi), 4),
                "boot_share_A_le_B": round(float((diffs <= 0).mean()), 4),
                "significant_95": bool(lo > 0 or hi < 0),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "analysis")
    args = ap.parse_args()

    data = {name: load_items(pats) for name, pats in SYSTEMS.items()}
    out_rows = []
    for a, b in PAIRS:
        out_rows += summarize(a, b, data[a], data[b], args.n_boot, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "bootstrap_comparisons.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    (args.out_dir / "bootstrap_comparisons.json").write_text(
        json.dumps({"n_boot": args.n_boot, "seed": args.seed, "rows": out_rows}, indent=2),
        encoding="utf-8",
    )
    for r in out_rows:
        if r["scope"] == "overall":
            print(
                f"{r['A']}  vs  {r['B']}\n  overall diff {r['diff_A_minus_B']:+.4f} "
                f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  significant={r['significant_95']}"
            )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
