#!/usr/bin/env python3
"""RouterBench-style comparison: solos, Rules v2, kNN, clf, Oracle, BestSingle + CIs.

  PYTHONPATH=. python scripts/build_router_comparison.py \\
    --details results/v6_eval/*_detail.jsonl \\
    --out results/v6_eval/router_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sample_size_ci import accuracy_ci  # noqa: E402

HEADLINE = ("knowledge", "translate", "alignment", "instruct")
SOLOS = ("mamay4", "mamay12", "lapa", "aya", "qwen7")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def index_details(paths: list[Path]) -> dict[str, dict[str, dict]]:
    """system -> id -> row"""
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in paths:
        for row in load_jsonl(path):
            sys = str(row.get("system") or "")
            iid = str(row.get("id") or "")
            if sys and iid:
                by[sys][iid] = row
    return by


def means(rows: list[dict], buckets: tuple[str, ...] | None = None) -> dict:
    filtered = [
        r
        for r in rows
        if buckets is None or str(r.get("bucket")) in buckets
    ]
    if not filtered:
        return {"n": 0, "quality": 0.0, "ci95": [0.0, 0.0]}
    by_b: dict[str, list[float]] = defaultdict(list)
    scores = []
    for r in filtered:
        s = float(r.get("score") or 0.0)
        scores.append(s)
        by_b[str(r.get("bucket") or "")] .append(s)
    micro = sum(scores) / len(scores)
    macros = [sum(v) / len(v) for v in by_b.values() if v]
    macro = sum(macros) / len(macros) if macros else 0.0
    lo, hi = accuracy_ci(micro, len(scores))
    return {
        "n": len(scores),
        "quality_micro": round(micro, 4),
        "quality_macro": round(macro, 4),
        "ci95_micro": [round(lo, 4), round(hi, 4)],
        "by_bucket": {b: round(sum(v) / len(v), 4) for b, v in sorted(by_b.items())},
    }


def oracle_rows(solo_maps: dict[str, dict[str, dict]]) -> list[dict]:
    ids = set()
    for mp in solo_maps.values():
        ids.update(mp)
    out = []
    for iid in ids:
        best = None
        winner = ""
        bucket = ""
        for sys, mp in solo_maps.items():
            row = mp.get(iid)
            if not row:
                continue
            bucket = str(row.get("bucket") or bucket)
            score = float(row.get("score") or 0.0)
            if best is None or score > best:
                best = score
                winner = sys
        if best is None:
            continue
        out.append({"id": iid, "bucket": bucket, "score": best, "system": "oracle", "winner": winner})
    return out


def markdown_table(systems: dict[str, dict]) -> str:
    lines = [
        "| System | n | micro | 95% CI | macro |",
        "|---|---:|---:|---|---:|",
    ]
    for name, stats in systems.items():
        ci = stats.get("ci95_micro") or [0, 0]
        lines.append(
            f"| {name} | {stats.get('n', 0)} | {stats.get('quality_micro', 0):.3f} "
            f"| [{ci[0]:.3f}, {ci[1]:.3f}] | {stats.get('quality_macro', 0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--details", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "router_comparison.json")
    p.add_argument("--headline-only", action="store_true")
    p.add_argument(
        "--lobo-report",
        type=Path,
        default=ROOT / "router" / "artifacts" / "train_report.json",
        help="Optional train_report.json; copies leave_one_bucket_out into the table payload.",
    )
    args = p.parse_args()

    by_sys = index_details(args.details)
    buckets = HEADLINE if args.headline_only else None
    solo_maps = {s: by_sys[s] for s in SOLOS if s in by_sys}
    systems: dict[str, dict] = {}
    for name, mp in sorted(by_sys.items()):
        systems[name] = means(list(mp.values()), buckets)

    if solo_maps:
        orows = oracle_rows(solo_maps)
        systems["oracle"] = means(orows, buckets)
        best_name = max(
            (n for n in solo_maps),
            key=lambda n: systems.get(n, {}).get("quality_macro", 0.0),
        )
        systems["best_single"] = {
            **systems[best_name],
            "alias": best_name,
        }

    lobo = None
    if args.lobo_report and args.lobo_report.exists():
        raw = json.loads(args.lobo_report.read_text(encoding="utf-8"))
        lobo = raw.get("leave_one_bucket_out")
    payload = {
        "headline_buckets": list(HEADLINE),
        "headline_only": args.headline_only,
        "systems": systems,
        "leave_one_bucket_out": lobo,
        "markdown": markdown_table(systems),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    md_path.write_text(payload["markdown"], encoding="utf-8")
    print(payload["markdown"])
    print(json.dumps({"wrote": str(args.out), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
