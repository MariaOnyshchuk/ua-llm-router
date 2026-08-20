#!/usr/bin/env python3
"""Sanity checks for router bucket leakage on mixed_ua_v4_balanced."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.intent_rules import route_intent  # noqa: E402

BENCH = ROOT / "benchmarks" / "mixed_ua_v4_balanced.jsonl"

# Previously leaked; must be fixed before combo bake-offs.
MUST_ROUTE = {
    "if-008": ("instruct", "mamay4"),
    "if-013": ("instruct", "mamay4"),
    "if-020": ("instruct", "mamay4"),
    "if-021": ("instruct", "mamay4"),
    "if-032": ("instruct", "mamay4"),
    "know-002": ("knowledge", "lapa"),
    "know-003": ("knowledge", "lapa"),
    "know-004": ("knowledge", "lapa"),
    "know-005": ("knowledge", "lapa"),
    "know-006": ("knowledge", "lapa"),
    "know-008": ("knowledge", "lapa"),
}


def load() -> list[dict]:
    rows = []
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def main() -> None:
    rows = {r["id"]: r for r in load()}
    failed = []
    for iid, (intent, model) in MUST_ROUTE.items():
        d = route_intent(rows[iid]["prompt"])
        if d.intent != intent or d.model != model:
            failed.append(f"{iid}: got {d.intent}/{d.model}, want {intent}/{model}")

    # overall gold agreement (chat may still disagree on soft cases)
    mismatches = []
    for r in rows.values():
        d = route_intent(r["prompt"])
        if r["bucket"] != d.intent:
            mismatches.append(f"{r['id']}: gold={r['bucket']} → {d.intent}/{d.model}")

    print(f"hard checks: {len(MUST_ROUTE) - len(failed)}/{len(MUST_ROUTE)} ok")
    for f in failed:
        print("  FAIL", f)
    print(f"gold≠intent remaining: {len(mismatches)}")
    for m in mismatches:
        print(" ", m)
    if failed:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
