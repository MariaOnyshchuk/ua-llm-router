#!/usr/bin/env python3
"""Summarize benchmark result JSONL files into a small comparison table.

  python scripts/summarize_results.py results/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args()
    rows = load_rows(args.files)
    if not rows:
        print("no rows", file=sys.stderr)
        sys.exit(1)

    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[str(r.get("system", "?"))].append(r)

    print(f"{'system':12} {'n':>4} {'ok':>4} {'avg_ms':>8} {'p50_ms':>8} {'p95_ms':>8}")
    for system, items in sorted(by_sys.items()):
        ok = [x for x in items if x.get("http_status") == 200 and x.get("content")]
        lats = sorted(float(x["latency_ms"]) for x in ok if x.get("latency_ms") is not None)
        avg = sum(lats) / len(lats) if lats else 0.0
        p50 = lats[len(lats) // 2] if lats else 0.0
        p95 = lats[min(len(lats) - 1, int(round(0.95 * (len(lats) - 1))))] if lats else 0.0
        print(f"{system:12} {len(items):4d} {len(ok):4d} {avg:8.1f} {p50:8.1f} {p95:8.1f}")

    # Routing distribution (router runs only)
    routed = [r for r in rows if r.get("router")]
    if routed:
        print("\nrouter decisions:")
        counts: dict[str, int] = defaultdict(int)
        for r in routed:
            counts[str((r.get("router") or {}).get("model", "?"))] += 1
        for model, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {model:8} {n}")


if __name__ == "__main__":
    main()
