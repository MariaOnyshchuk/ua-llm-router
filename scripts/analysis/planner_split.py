#!/usr/bin/env python3
"""How often does the LLM planner split a single-skill request into several steps?

Every v6 item is one skill (its bucket). The right plan for it is ONE step whose intent
equals the bucket. This script sends only the planner call (no specialist execution) to
one model and counts what comes back:

  split      valid plan with more than one step (over-planning)
  fallback   planner output unusable, the 1-step chat fallback plan is used
  correct    a valid plan with one step whose intent matches the bucket

A stratified random sample per bucket is used (seed 42). Intervals are Wilson 95%.
Item rows keep ids and counts only, never prompts or model text (v6 reprints ZNO,
FLORES, UAlign and other source text; see AGENTS.md).

  python scripts/analysis/planner_split.py --n-per-bucket 100
  python scripts/analysis/planner_split.py --profile hybrid_mamay12 --n-per-bucket 100

Needs the planner model on its usual port (mamay4 :8003, through the SSH tunnel).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLAIM_BUCKETS = ("knowledge", "translate", "alignment", "instruct")
Caller = Callable[[str, str], Awaitable[dict[str, Any]]]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sample_items(items: list[dict], per_bucket: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if item.get("bucket") in CLAIM_BUCKETS:
            by[item["bucket"]].append(item)
    out: list[dict] = []
    for bucket in CLAIM_BUCKETS:
        pool = sorted(by[bucket], key=lambda x: x["id"])
        out += pool if per_bucket <= 0 else rng.sample(pool, min(per_bucket, len(pool)))
    return out


async def plan_one(item: dict, caller: Caller, profile: str) -> dict[str, Any]:
    from router.planner import generate_plan

    plan, planning = await generate_plan(item["prompt"], caller, profile=profile)
    attempts = planning.get("attempts") or []
    intents = [s.intent for s in plan.steps]
    valid = bool(planning.get("valid")) and not planning.get("fallback")
    return {
        "id": item["id"],
        "bucket": item["bucket"],
        "valid": valid,
        "fallback": bool(planning.get("fallback")),
        "repaired": bool(planning.get("repaired")),
        "n_steps": len(plan.steps),
        "intents": intents,
        "planner_calls": len(attempts),
        "latency_ms": round(sum(float(a.get("latency_ms") or 0) for a in attempts), 1),
        "truncated": any(a.get("finish_reason") == "length" for a in attempts),
        "split": valid and len(plan.steps) > 1,
        "correct": valid and len(plan.steps) == 1 and intents[0] == item["bucket"],
    }


async def run(items: list[dict], caller: Caller, profile: str, concurrency: int, progress: bool = False) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(item: dict) -> dict:
        nonlocal done
        async with sem:
            row = await plan_one(item, caller, profile)
        done += 1
        if progress and done % 20 == 0:
            print(f"  {done}/{len(items)}", file=sys.stderr, flush=True)
        return row

    return await asyncio.gather(*[one(i) for i in items])


def summarise(rows: list[dict]) -> dict[str, Any]:
    def block(sub: list[dict]) -> dict[str, Any]:
        n = len(sub)
        out: dict[str, Any] = {"n": n}
        for key in ("split", "fallback", "correct", "repaired", "truncated"):
            k = sum(bool(r[key]) for r in sub)
            lo, hi = wilson(k, n)
            out[key] = {"k": k, "rate": round(k / n, 4) if n else None, "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        out["steps"] = dict(sorted(Counter(r["n_steps"] for r in sub).items()))
        out["latency_p50_ms"] = sorted(r["latency_ms"] for r in sub)[n // 2] if n else None
        return out

    result = {"overall": block(rows), "by_bucket": {b: block([r for r in rows if r["bucket"] == b]) for b in CLAIM_BUCKETS}}
    other = Counter(i for r in rows if r["split"] for i in r["intents"])
    result["intents_in_split_plans"] = dict(other.most_common())
    return result


def print_table(summary: dict[str, Any], label: str) -> None:
    print(f"\nPlanner: {label}; right answer for every item = 1 step with intent == bucket\n")
    print("| bucket | n | split (>1 step) | fallback | correct | truncated |")
    print("|---|---|---|---|---|---|")
    for name, s in [*summary["by_bucket"].items(), ("all", summary["overall"])]:
        cell = lambda k: f"{s[k]['rate']:.2f} [{s[k]['ci_low']:.2f}, {s[k]['ci_high']:.2f}]"
        print(f"| {name} | {s['n']} | {cell('split')} | {cell('fallback')} | {cell('correct')} | {cell('truncated')} |")
    print("\nsteps per plan (all):", summary["overall"]["steps"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default="benchmarks/mixed_ua_v6.jsonl")
    p.add_argument("--profile", default="hybrid")
    p.add_argument("--n-per-bucket", type=int, default=100, help="0 = every item")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=256, help="planner budget; composite runs use 256")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out-prefix", default="results/analysis/planner_split_v6")
    args = p.parse_args()

    import httpx
    from router.client import async_chat
    from router.planner import PLANNER_PROFILES, resolve_profile

    if args.profile not in PLANNER_PROFILES or args.profile == "template":
        raise SystemExit(f"use an LLM planner profile: {[x for x in PLANNER_PROFILES if x != 'template']}")
    _, planner_alias = resolve_profile(args.profile)
    planner_alias = planner_alias or "mamay4"

    items = [json.loads(x) for x in (ROOT / args.bench).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    sample = sample_items(items, args.n_per_bucket, args.seed)
    print(f"{len(sample)} items, planner {args.profile} on {planner_alias}, max_tokens {args.max_tokens}", file=sys.stderr)

    async def go() -> list[dict]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async def caller(alias: str, prompt: str) -> dict[str, Any]:
                return await async_chat(client, alias, prompt, temperature=0.0, max_tokens=args.max_tokens, seed=42)
            probe = await caller(planner_alias, "ok")
            if not probe.get("ok"):
                raise SystemExit(f"{planner_alias} not reachable: {probe.get('error')}")
            return await run(sample, caller, args.profile, args.concurrency, progress=True)

    rows = asyncio.run(go())
    summary = summarise(rows)
    summary["settings"] = {"profile": args.profile, "planner": planner_alias, "n_per_bucket": args.n_per_bucket,
                           "seed": args.seed, "max_tokens": args.max_tokens, "bench": args.bench, "temperature": 0.0}
    prefix = ROOT / f"{args.out_prefix}_{args.profile}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(f"{prefix}_items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print_table(summary, f"{args.profile} ({planner_alias}), max_tokens {args.max_tokens}")
    print(f"\nwrote {prefix}.json and {prefix}_items.jsonl")


if __name__ == "__main__":
    main()
