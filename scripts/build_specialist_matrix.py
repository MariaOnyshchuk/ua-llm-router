#!/usr/bin/env python3
"""Build a specialist bake-off matrix for router decisions.

Before wiring a combination into the router, compare single models on the SAME
suite. Route bucket B → model X only if X beats the default (usually Mamay-4B)
on B by a clear margin.

Inputs: one or more score JSON files from score_results.py
        OR scores_mean_sd_v3.json from aggregate_repeat_scores.py

Example:
  python scripts/build_specialist_matrix.py \\
    --scores results/scores_mean_sd_v3.json \\
    --default mamay4 \\
    --margin 0.02 \\
    --out results/specialist_matrix_v3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BUCKETS = ("chat", "translate", "instruct", "knowledge", "code", "alignment")


def _quality(sys_block: dict[str, Any], bucket: str) -> float | None:
    # mean±sd aggregate file
    key = f"bucket.{bucket}.quality"
    if key in sys_block and isinstance(sys_block[key], dict) and "mean" in sys_block[key]:
        return float(sys_block[key]["mean"])
    # plain score_results file
    bb = (sys_block.get("by_bucket") or {}).get(bucket)
    if bb and "quality_mean" in bb:
        return float(bb["quality_mean"])
    return None


def _latency_p50(sys_block: dict[str, Any], bucket: str) -> float | None:
    key = f"bucket.{bucket}.latency_p50"
    if key in sys_block and isinstance(sys_block[key], dict) and "mean" in sys_block[key]:
        return float(sys_block[key]["mean"])
    bb = (sys_block.get("by_bucket") or {}).get(bucket)
    if bb and "latency_p50_ms" in bb:
        return float(bb["latency_p50_ms"])
    return None


def _overall(sys_block: dict[str, Any], which: str) -> float | None:
    # which: overall_mean | overall_mean_macro
    if which in sys_block and isinstance(sys_block[which], dict) and "mean" in sys_block[which]:
        return float(sys_block[which]["mean"])
    if which in sys_block and isinstance(sys_block[which], (int, float)):
        return float(sys_block[which])
    return None


def load_systems(paths: list[Path]) -> dict[str, dict[str, Any]]:
    systems: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("systems") or {}
        for name, s in block.items():
            # Prefer later files / keep last
            systems[name] = s
    return systems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", nargs="+", type=Path, required=True)
    p.add_argument("--default", default="mamay4", help="Default / S0 model for deltas")
    p.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Min quality gain over default to recommend routing that bucket",
    )
    p.add_argument("--out", type=Path, default=Path("results/specialist_matrix.json"))
    args = p.parse_args()

    systems = load_systems(args.scores)
    if args.default not in systems:
        raise SystemExit(
            f"default '{args.default}' not in systems: {sorted(systems)}"
        )

    # quality matrix: bucket -> model -> score
    matrix: dict[str, dict[str, float | None]] = {b: {} for b in BUCKETS}
    latency: dict[str, dict[str, float | None]] = {b: {} for b in BUCKETS}
    for name, s in systems.items():
        for b in BUCKETS:
            matrix[b][name] = _quality(s, b)
            latency[b][name] = _latency_p50(s, b)

    # winners + route advice
    advice: dict[str, Any] = {}
    for b in BUCKETS:
        scored = {m: q for m, q in matrix[b].items() if q is not None}
        if not scored:
            advice[b] = {"winner": None, "route_to": args.default, "reason": "no scores"}
            continue
        winner = max(scored, key=lambda m: scored[m])
        base = scored.get(args.default)
        gain = None if base is None else scored[winner] - base
        # Prefer non-default only if clear gain
        route_to = args.default
        reason = f"keep {args.default}"
        if winner != args.default and base is not None and (scored[winner] - base) >= args.margin:
            route_to = winner
            reason = f"{winner} beats {args.default} by {scored[winner] - base:+.3f} (≥{args.margin})"
        elif winner == args.default:
            reason = f"{args.default} wins bucket"
        else:
            reason = (
                f"{winner} highest but gain "
                f"{0.0 if gain is None else gain:+.3f} < margin {args.margin} → keep {args.default}"
            )
        # deltas vs default
        deltas = {
            m: (None if base is None or q is None else round(q - base, 4))
            for m, q in matrix[b].items()
        }
        advice[b] = {
            "winner": winner,
            "winner_quality": round(scored[winner], 4),
            "default_quality": None if base is None else round(base, 4),
            "gain_vs_default": None if gain is None else round(gain, 4),
            "route_to": route_to,
            "reason": reason,
            "deltas_vs_default": deltas,
        }

    overall = {}
    for name, s in systems.items():
        overall[name] = {
            "micro": _overall(s, "overall_mean"),
            "macro": _overall(s, "overall_mean_macro"),
        }

    # Suggested router sketch from advice
    routes = {b: advice[b]["route_to"] for b in BUCKETS}
    unique_targets = sorted({r for r in routes.values() if r})
    nontrivial = {b: t for b, t in routes.items() if t != args.default}

    out = {
        "default": args.default,
        "margin": args.margin,
        "models": sorted(systems),
        "overall": overall,
        "quality_by_bucket": matrix,
        "latency_p50_by_bucket": latency,
        "bucket_advice": advice,
        "suggested_routes": routes,
        "nondefault_routes": nontrivial,
        "router_worth_building": bool(nontrivial),
        "summary": (
            f"Route {len(nontrivial)}/{len(BUCKETS)} buckets away from {args.default}: "
            f"{nontrivial or 'none — single-model S0 is enough for now'}"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human table
    models = sorted(systems)
    print(f"default={args.default}  margin={args.margin}")
    print(f"{'bucket':12}", *[f"{m:>14}" for m in models], "  advice")
    for b in BUCKETS:
        cells = []
        for m in models:
            q = matrix[b].get(m)
            cells.append(f"{q:14.4f}" if q is not None else f"{'—':>14}")
        a = advice[b]
        print(f"{b:12}", *cells, f"  → {a['route_to']} ({a['reason']})")
    print()
    print("overall:")
    for m in models:
        o = overall[m]
        print(f"  {m:16} micro={o['micro']}  macro={o['macro']}")
    print()
    print(out["summary"])
    print(json.dumps({"wrote": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
