#!/usr/bin/env python3
"""Routing / cascade counts from a result JSONL (one file or glob)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    n = 0
    http_ok = 0
    models = Counter()
    intents = Counter()
    escalate = 0
    escalate_ids: list[str] = []
    prompt_variants = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        if int(row.get("http_status") or 0) == 200:
            http_ok += 1
        router = row.get("router") or {}
        models[str(router.get("model") or row.get("model_requested") or "?")] += 1
        intents[str(router.get("intent") or "?")] += 1
        prompt_variants[str(row.get("prompt_variant") or "")] += 1
        casc = row.get("cascade") or {}
        if casc.get("escalated"):
            escalate += 1
            escalate_ids.append(str(row.get("id")))
    return {
        "file": str(path),
        "n": n,
        "http_ok": http_ok,
        "models": dict(models),
        "intents": dict(intents),
        "prompt_variant": dict(prompt_variants),
        "escalated": escalate,
        "escalate_rate": round(escalate / max(1, n), 4),
        "escalate_ids": escalate_ids,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rows = [summarize(p) for path in args.paths for p in ([path] if path.is_file() else path.glob("*.jsonl"))]
    # skip meta
    rows = [r for r in rows if not r["file"].endswith(".meta.json") and r["n"]]
    print(json.dumps(rows if len(rows) != 1 else rows[0], ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
