#!/usr/bin/env python3
"""Ask the two AgentCoMa-UK step questions alone, one model, and score them.

Each item has a commonsense question and a math question that, chained, make the
composite task. This script sends each question to one model on its own so the
compositional gap can be computed:

    gap = P(both steps right alone) - P(composite final answer right)

    run:    python scripts/run_agentcoma_steps.py run --alias mamay12 --split dev
    score:  python scripts/run_agentcoma_steps.py score results/.../steps_*.jsonl

Rows hold gated prompts: keep the output local (see .gitignore), share scores only.
Rubrics are the same as the composite oracle plan (contains_any / numeric).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from router.backends import BACKENDS  # noqa: E402
from router.client import async_chat  # noqa: E402
from score_composite_results import mean, score_rubric  # noqa: E402

MERGED = "benchmarks/agentcoma_uk_200_merged.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


async def run(args: argparse.Namespace) -> None:
    if args.alias not in BACKENDS:
        raise SystemExit(f"unknown alias {args.alias}")
    items = load_jsonl(ROOT / args.bench)
    if args.split:
        allowed = {x.strip() for x in args.split.split(",")}
        items = [x for x in items if x.get("split") in allowed]
    if args.limit:
        items = items[: args.limit]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{args.alias}_steps_agentcoma200_t0_s{args.seed}_{stamp}.jsonl"
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        with path.open("w", encoding="utf-8") as out:
            for index, item in enumerate(items, 1):
                row = {"id": item["id"], "system": args.alias, "split": item.get("split"),
                       "category": item.get("category"), "operation_type": item.get("operation_type")}
                for step, key in (("commonsense", "question_commonsense_uk"), ("math", "question_math_uk")):
                    hop = await async_chat(client, args.alias, item[key], temperature=0.0,
                                           max_tokens=args.max_tokens, seed=args.seed)
                    row[step] = {"prompt": item[key], "content": hop.get("content") or "",
                                 "ok": bool(hop.get("ok")), "latency_ms": hop.get("latency_ms"),
                                 "gpu_seconds": hop.get("gpu_seconds")}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{args.alias} {index}/{len(items)}] {item['id']}")
    print(f"wrote {path}")


def score(args: argparse.Namespace) -> None:
    gold = {x["id"]: x for x in load_jsonl(ROOT / args.bench)}
    result: dict[str, dict] = {}
    detail = []
    for name in args.files:
        for row in load_jsonl(Path(name)):
            item = gold[row["id"]]
            cs, _ = score_rubric(row["commonsense"]["content"],
                                 {"type": "contains_any", "values": item["answers_commonsense_uk"]})
            ma, _ = score_rubric(row["math"]["content"],
                                 {"type": "numeric", "value": float(item["answer_math"])})
            ok = bool(row["commonsense"]["ok"] and row["math"]["ok"])
            detail.append({"system": row["system"], "id": row["id"], "split": row.get("split"),
                           "category": row.get("category"), "operation_type": row.get("operation_type"),
                           "commonsense": cs, "math": ma, "both": float(cs == 1.0 and ma == 1.0),
                           "http_ok": ok})
    for system in sorted({d["system"] for d in detail}):
        rows = [d for d in detail if d["system"] == system]
        result[system] = {"n": len(rows), "http_failures": sum(not r["http_ok"] for r in rows),
                          "commonsense": round(mean([r["commonsense"] for r in rows]), 4),
                          "math": round(mean([r["math"] for r in rows]), 4),
                          "both": round(mean([r["both"] for r in rows]), 4)}
    out = Path(args.out)
    out.write_text(json.dumps({"systems": result}, indent=2, ensure_ascii=False), encoding="utf-8")
    out.with_name(out.stem + "_detail.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in detail), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--alias", required=True)
    r.add_argument("--bench", default=MERGED)
    r.add_argument("--split", default="")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--timeout", type=float, default=180.0)
    r.add_argument("--out-dir", default="results/week_13_agentcoma_steps")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--bench", default=MERGED)
    s.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    else:
        score(args)


if __name__ == "__main__":
    main()
