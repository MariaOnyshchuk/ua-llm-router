#!/usr/bin/env python3
"""Merge base + extra JSONL benchmark files; drop duplicate ids (first wins)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    seen: set[str] = set()
    out_rows: list[dict] = []
    sources = [args.base, *args.extra]
    for path in sources:
        for row in load_jsonl(path):
            rid = str(row.get("id"))
            if rid in seen:
                continue
            seen.add(rid)
            out_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("# Merged benchmark suite — generated; do not hand-edit\n")
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_bucket: dict[str, int] = {}
    for row in out_rows:
        b = row.get("bucket") or "?"
        by_bucket[b] = by_bucket.get(b, 0) + 1
    print(json.dumps({"wrote": str(args.out), "n": len(out_rows), "by_bucket": by_bucket}, indent=2))


if __name__ == "__main__":
    main()
