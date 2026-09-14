#!/usr/bin/env python3
"""IRT-inspired curation: keep high between-model score variance, then balance difficulty.

Usage:
  python scripts/curate_discriminative_suite.py \\
    --suite benchmarks/mixed_ua_v6_screen.jsonl \\
    --scores results/v6_screen/scores_*_detail.jsonl \\
    --out benchmarks/mixed_ua_v6.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sample_size_ci import binomial_n  # noqa: E402
HEADLINE = ("knowledge", "translate", "alignment", "instruct")
APPENDIX = ("chat", "code")
ALL_BUCKETS = HEADLINE + APPENDIX


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def collect_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    """item_id -> {system: score} (last write wins)."""
    by_item: dict[str, dict[str, float]] = defaultdict(dict)
    for path in paths:
        for row in load_jsonl(path):
            iid = str(row.get("id") or "")
            sys = str(row.get("system") or "")
            if not iid or not sys:
                continue
            by_item[iid][sys] = float(row.get("score") or 0.0)
    return by_item


def proxy_stats(scores: dict[str, float]) -> tuple[float, float]:
    vals = list(scores.values())
    if not vals:
        return (0.0, 0.0)
    mean = statistics.fmean(vals)
    var = statistics.pvariance(vals) if len(vals) > 1 else 0.0
    return (var, mean)


def difficulty_balanced(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n >= len(rows):
        return rows[:]
    ranked = sorted(rows, key=lambda r: float(r["_mean"]))
    cuts = [0, len(ranked) // 3, (2 * len(ranked)) // 3, len(ranked)]
    bands = [ranked[cuts[i] : cuts[i + 1]] for i in range(3)]
    per = max(1, n // 3)
    chosen: list[dict] = []
    leftover: list[dict] = []
    for band in bands:
        chunk = band[:]
        rng.shuffle(chunk)
        chosen.extend(chunk[:per])
        leftover.extend(chunk[per:])
    rng.shuffle(leftover)
    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
    else:
        chosen.extend(leftover[: n - len(chosen)])
    return chosen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--scores", nargs="+", type=Path, required=True)
    p.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Aliases used for variance/difficulty (default: all). "
        "Pass mamay4 lapa aya to drop a broken screening run.",
    )
    p.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "mixed_ua_v6.jsonl")
    p.add_argument("--meta-out", type=Path, default=None)
    p.add_argument("--drop-frac", type=float, default=0.43)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p", type=float, default=0.8)
    p.add_argument("--epsilon", type=float, default=0.03)
    p.add_argument("--z", type=float, default=1.96)
    p.add_argument(
        "--keep-appendix",
        action="store_true",
        help="Also keep chat/code (whole remaining pool after drop).",
    )
    args = p.parse_args()
    if args.out.resolve() in {
        (ROOT / "benchmarks" / "mixed_ua_v4_balanced.jsonl").resolve(),
        (ROOT / "benchmarks" / "mixed_ua_v5.jsonl").resolve(),
        (ROOT / "benchmarks" / "mixed_ua_v3.jsonl").resolve(),
    }:
        raise SystemExit("refusing to overwrite a historical suite")

    items = {str(r.get("id")): r for r in load_jsonl(args.suite)}
    scored = collect_scores(args.scores)
    if args.systems:
        allowed = set(args.systems)
        scored = {
            iid: {sys: val for sys, val in model_scores.items() if sys in allowed}
            for iid, model_scores in scored.items()
        }
    n_star = binomial_n(p=args.p, epsilon=args.epsilon, z=args.z)
    rng = random.Random(args.seed)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    skipped_unscored = 0
    for iid, item in items.items():
        bucket = str(item.get("bucket") or "")
        if bucket not in ALL_BUCKETS:
            continue
        model_scores = scored.get(iid) or {}
        if len(model_scores) < 2:
            skipped_unscored += 1
            continue
        var, mean = proxy_stats(model_scores)
        rec = dict(item)
        rec["_var"] = var
        rec["_mean"] = mean
        rec["_n_models"] = len(model_scores)
        by_bucket[bucket].append(rec)

    out_rows: list[dict] = []
    bucket_meta: dict[str, dict] = {}
    for bucket in ALL_BUCKETS:
        pool = by_bucket[bucket]
        if not pool:
            bucket_meta[bucket] = {"selected": 0, "pool": 0, "note": "empty"}
            continue
        pool.sort(key=lambda r: (-float(r["_var"]), str(r.get("id"))))
        drop_n = int(math.floor(len(pool) * args.drop_frac))
        kept = pool[drop_n:] if drop_n < len(pool) else pool[:]
        headline = bucket in HEADLINE
        target = n_star if headline else len(kept)
        if bucket in APPENDIX and not args.keep_appendix:
            selected = []
            note = "appendix excluded from headline v6 (pass --keep-appendix)"
        else:
            selected = difficulty_balanced(kept, target, rng)
            selected.sort(key=lambda r: str(r.get("id")))
            note = "headline" if headline else "appendix"
        clean = []
        for row in selected:
            item = {k: v for k, v in row.items() if not str(k).startswith("_")}
            clean.append(item)
        out_rows.extend(clean)
        bucket_meta[bucket] = {
            "pool_scored": len(pool),
            "dropped_low_var": drop_n if bucket not in APPENDIX or args.keep_appendix else 0,
            "after_drop": len(kept),
            "selected": len(clean),
            "target_n": target if headline else len(kept),
            "n_star": n_star,
            "mean_var": round(statistics.fmean(float(r["_var"]) for r in pool), 6),
            "role": note,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f"# mixed_ua_v6 curated — seed={args.seed} drop_frac={args.drop_frac} "
            f"n*={n_star} ε={args.epsilon}; do not hand-edit\n"
        )
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "suite": str(args.suite),
        "scores": [str(x) for x in args.scores],
        "wrote": str(args.out),
        "n": len(out_rows),
        "n_star": n_star,
        "p": args.p,
        "epsilon": args.epsilon,
        "z": args.z,
        "formula": "N >= (z * sqrt(p*(1-p)) / epsilon)^2",
        "drop_frac": args.drop_frac,
        "skipped_unscored": skipped_unscored,
        "systems": list(args.systems) if args.systems else "all",
        "headline_buckets": list(HEADLINE),
        "appendix_buckets": list(APPENDIX),
        "by_bucket": bucket_meta,
    }
    meta_path = args.meta_out or args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
