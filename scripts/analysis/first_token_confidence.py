#!/usr/bin/env python3
"""Collect first-token probabilities for closed-answer items (knowledge, alignment).

For each item and model: the top tokens at the first generated position and
their probabilities. Used to test a confidence cascade ("keep the first model's
answer when it is sure, otherwise ask another model"). Writes ids, tokens and
probabilities only; no prompt or answer text.

    PYTHONPATH=. python scripts/analysis/first_token_confidence.py \
        --split dev --systems lapa,mamay4,aya,mamay12 \
        --out results/v7_confidence/dev_first_token.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import httpx

from router.backends import BACKENDS

ROOT = Path(__file__).resolve().parents[2]


async def one(client: httpx.AsyncClient, sem: asyncio.Semaphore, alias: str, item: dict[str, Any],
              top: int) -> dict[str, Any]:
    base, model = BACKENDS[alias][0], BACKENDS[alias][1]
    body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0, "seed": 42, "max_tokens": 1, "logprobs": True, "top_logprobs": top}
    async with sem:
        try:
            resp = await client.post(f"{base}/chat/completions", json=body)
            resp.raise_for_status()
            first = resp.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            tokens = [[t["token"], round(math.exp(t["logprob"]), 5)] for t in first]
            return {"system": alias, "id": item["id"], "bucket": item["bucket"], "http_status": 200, "top": tokens}
        except Exception as exc:  # noqa: BLE001
            return {"system": alias, "id": item["id"], "bucket": item["bucket"], "http_status": 0,
                    "error": type(exc).__name__, "top": []}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="benchmarks/mixed_ua_v7.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--buckets", default="knowledge,alignment")
    parser.add_argument("--systems", default="lapa,mamay4,aya")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    buckets = set(args.buckets.split(","))
    items = [json.loads(line) for line in (ROOT / args.bench).read_text(encoding="utf-8").splitlines()
             if line.startswith("{")]
    items = [r for r in items if r.get("split") == args.split and r["bucket"] in buckets]
    systems = args.systems.split(",")
    sems = {alias: asyncio.Semaphore(args.concurrency) for alias in systems}
    async with httpx.AsyncClient(timeout=300) as client:
        rows = await asyncio.gather(*[one(client, sems[a], a, it, args.top) for a in systems for it in items])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for alias in systems:
        mine = [r for r in rows if r["system"] == alias]
        print(alias, len(mine), "http 200:", sum(r["http_status"] == 200 for r in mine))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
