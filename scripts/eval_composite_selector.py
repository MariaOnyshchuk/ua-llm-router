#!/usr/bin/env python3
"""Offline evaluation of constrained template selection (no GPU).

Measures exact workflow match / template-id accuracy against oracle plans on
mixed_ua_composite_v2. Intended for train/dev ablations before any live bake-off.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.orchestrator import Plan, plan_signature  # noqa: E402
from router.workflows import select_plan, select_template  # noqa: E402
from scripts.build_composite_benchmark import build_items_v2, validate_items_v2  # noqa: E402


def load_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        items = build_items_v2()
        validate_items_v2(items)
        return items
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def evaluate(items: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        oracle = Plan.from_dict(item["oracle_plan"], source="oracle")
        plan, meta = select_plan(item["prompt"])
        template = select_template(item["prompt"])
        match = plan_signature(plan) == plan_signature(oracle)
        expected_tid = item.get("template_id") or item.get("family")
        rows.append(
            {
                "id": item["id"],
                "split": item.get("split"),
                "family": item["family"],
                "template_id_expected": expected_tid,
                "template_id_selected": meta.get("template_id") or (template.id if template else None),
                "plan_match": match,
                "fallback": bool(meta.get("fallback")),
                "fallback_reason": meta.get("fallback_reason"),
                "superfluous_steps": max(0, len(plan.steps) - len(oracle.steps)),
                "missing_steps": max(0, len(oracle.steps) - len(plan.steps)),
            }
        )

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "plan_exact_match_rate": round(
                sum(r["plan_match"] for r in subset) / len(subset), 4
            ),
            "template_id_accuracy": round(
                sum(
                    r["template_id_selected"] == r["template_id_expected"] for r in subset
                )
                / len(subset),
                4,
            ),
            "fallback_rate": round(sum(r["fallback"] for r in subset) / len(subset), 4),
            "mean_superfluous_steps": round(
                sum(r["superfluous_steps"] for r in subset) / len(subset), 4
            ),
            "mean_missing_steps": round(
                sum(r["missing_steps"] for r in subset) / len(subset), 4
            ),
            "misses": [
                r["id"]
                for r in subset
                if not r["plan_match"] or r["template_id_selected"] != r["template_id_expected"]
            ],
        }

    by_split: dict[str, Any] = {}
    for split in sorted({r["split"] for r in rows if r["split"]}):
        by_split[str(split)] = summarize([r for r in rows if r["split"] == split])
    by_family: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        by_family[str(family)] = summarize([r for r in rows if r["family"] == family])

    return {
        "overall": summarize(rows),
        "by_split": by_split,
        "by_family": by_family,
        "error_counts": dict(
            Counter(
                "fallback"
                if r["fallback"]
                else ("wrong_template" if r["template_id_selected"] != r["template_id_expected"] else "signature_mismatch")
                for r in rows
                if not r["plan_match"]
            )
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        type=Path,
        default=ROOT / "benchmarks" / "mixed_ua_composite_v2.jsonl",
    )
    parser.add_argument("--split", default="", help="train,dev,test or comma list")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "week_12_composite_v2_selector" / "selector_offline.json",
    )
    args = parser.parse_args()
    items = load_items(args.bench if args.bench.exists() else None)
    if args.split:
        allowed = {x.strip() for x in args.split.split(",") if x.strip()}
        items = [i for i in items if i.get("split") in allowed]
    report = evaluate(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Keep detail rows in a sibling file; summary stays small for docs.
    detail_path = args.out.with_name(args.out.stem + "_detail.jsonl")
    summary = {k: v for k, v in report.items() if k != "rows"}
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report["rows"]),
        encoding="utf-8",
    )
    overall = summary["overall"]
    print(
        f"n={overall['n']} exact={overall['plan_exact_match_rate']:.3f} "
        f"template_acc={overall['template_id_accuracy']:.3f} "
        f"fallback={overall['fallback_rate']:.3f}"
    )
    for split, values in summary["by_split"].items():
        print(
            f"  {split:5} n={values['n']} exact={values['plan_exact_match_rate']:.3f} "
            f"tid={values['template_id_accuracy']:.3f}"
        )
    print(f"wrote {args.out}")
    if overall.get("misses"):
        print(f"misses ({len(overall['misses'])}): {', '.join(overall['misses'][:12])}")


if __name__ == "__main__":
    main()
