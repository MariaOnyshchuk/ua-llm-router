#!/usr/bin/env python3
"""Score every *rep*.jsonl in a directory and aggregate mean±sd."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pattern", default="*_rep*.jsonl")
    args = p.parse_args()
    d = args.dir if args.dir.is_absolute() else ROOT / args.dir
    files = sorted(x for x in d.glob(args.pattern) if x.suffix == ".jsonl" and ".meta" not in x.name)
    if not files:
        raise SystemExit(f"No JSONL matching {args.pattern} in {d}")
    scored_dir = d / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    score_jsons = []
    for f in files:
        out = scored_dir / (f.stem + ".scores.json")
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "score_results.py"), str(f), "--out", str(out)]
        )
        score_jsons.append(out)
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "aggregate_repeat_scores.py"), *[str(x) for x in score_jsons], "--out", str(args.out)]
    )


if __name__ == "__main__":
    main()
