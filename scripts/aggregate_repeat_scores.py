#!/usr/bin/env python3
"""Aggregate repeated scored runs into mean ± sd.

Example:
  python scripts/score_results.py results/mamay4_*_rep1.jsonl --out results/s_rep1.json
  python scripts/score_results.py results/mamay4_*_rep2.jsonl --out results/s_rep2.json
  python scripts/score_results.py results/mamay4_*_rep3.jsonl --out results/s_rep3.json
  python scripts/aggregate_repeat_scores.py results/s_rep*.json --out results/scores_mean_sd.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean_sd(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/scores_mean_sd.json"))
    args = p.parse_args()

    # system -> metric_path -> list of values across repeats
    series: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_files = 0
    for path in args.files:
        data = json.loads(path.read_text(encoding="utf-8"))
        n_files += 1
        for sys, s in (data.get("systems") or {}).items():
            series[sys]["overall_mean"].append(float(s.get("overall_mean", 0)))
            if "overall_mean_macro" in s:
                series[sys]["overall_mean_macro"].append(float(s["overall_mean_macro"]))
            series[sys]["latency_avg_ms"].append(float(s.get("latency_avg_ms", 0)))
            series[sys]["latency_p50_ms"].append(float(s.get("latency_p50_ms", 0)))
            for bucket, b in (s.get("by_bucket") or {}).items():
                series[sys][f"bucket.{bucket}.quality"].append(float(b.get("quality_mean", 0)))
                series[sys][f"bucket.{bucket}.latency_p50"].append(float(b.get("latency_p50_ms", 0)))

    out: dict[str, Any] = {"n_repeats": n_files, "systems": {}}
    print(f"{'system':12} {'metric':28} {'mean':>8} {'sd':>8} {'n':>3}")
    for sys in sorted(series):
        out["systems"][sys] = {}
        for metric, vals in sorted(series[sys].items()):
            m, sd = mean_sd(vals)
            out["systems"][sys][metric] = {
                "mean": round(m, 4),
                "sd": round(sd, 4),
                "n": len(vals),
                "values": [round(v, 4) for v in vals],
            }
            print(f"{sys:12} {metric:28} {m:8.4f} {sd:8.4f} {len(vals):3d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
