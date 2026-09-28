#!/usr/bin/env python3
"""Run direct and orchestrated systems on composite benchmarks (v1/v2)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.backends import BACKENDS  # noqa: E402
from router.client import async_chat  # noqa: E402
from router.intent_rules import route_intent  # noqa: E402
from router.orchestrator import Plan, execute_plan, plan_signature  # noqa: E402
from router.planner import PLANNER_PROFILES, generate_plan  # noqa: E402

DIRECT_SYSTEMS = ("mamay4", "lapa", "aya", "qwen7", "mamay12")
DEFAULT_SYSTEMS = ("mamay4", "lapa", "aya", "qwen7", "router_direct", "oracle", "hybrid")
ORCHESTRATED = frozenset({"router_direct", "oracle", *PLANNER_PROFILES})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


async def health_check(aliases: set[str]) -> None:
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for alias in sorted(aliases):
            base, _ = BACKENDS[alias]
            try:
                response = await client.get(f"{base}/models")
                if response.status_code != 200:
                    errors.append(f"{alias}: HTTP {response.status_code}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{alias}: {type(exc).__name__}: {exc}")
    if errors:
        raise SystemExit("Composite preflight failed:\n  - " + "\n  - ".join(errors))


def required_aliases(systems: list[str]) -> set[str]:
    required: set[str] = set()
    for system in systems:
        if system in BACKENDS:
            required.add(system)
        if system in {"router_direct", "oracle"} or system in PLANNER_PROFILES:
            required.update({"mamay4", "lapa", "aya"})
        if system == "hybrid_mamay12":
            required.add("mamay12")
    return required


def balanced_limit(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin families so a small smoke never covers only the first family."""
    if limit <= 0 or limit >= len(items):
        return items
    families: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        families.setdefault(str(item.get("family")), []).append(item)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(families.values()):
        for family in sorted(families):
            if families[family] and len(selected) < limit:
                selected.append(families[family].pop(0))
    return selected


def filter_items(
    items: list[dict[str, Any]],
    *,
    family: str,
    split: str,
    limit: int,
) -> list[dict[str, Any]]:
    if family:
        allowed = {x.strip() for x in family.split(",") if x.strip()}
        items = [item for item in items if item.get("family") in allowed]
    if split:
        allowed_splits = {x.strip() for x in split.split(",") if x.strip()}
        items = [item for item in items if item.get("split") in allowed_splits]
    if limit:
        items = balanced_limit(items, limit)
    return items


async def run_item(
    item: dict[str, Any],
    system: str,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = str(item["prompt"])

    async def caller(alias: str, sub_prompt: str) -> dict[str, Any]:
        return await async_chat(
            client,
            alias,
            sub_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

    oracle = Plan.from_dict(item["oracle_plan"], source="oracle")
    plan: Plan | None = None
    planning: dict[str, Any] | None = None
    planner_profile: str | None = None

    if system in BACKENDS:
        hop = await caller(system, prompt)
        execution = {
            "steps": [],
            "complete": bool(hop.get("ok")),
            "final_output": hop.get("content") or "",
            "final_step": "direct",
            "calls": 1,
            "latency_ms": hop.get("latency_ms") or 0.0,
            "gpu_seconds": hop.get("gpu_seconds") or 0.0,
            "direct": hop,
        }
    elif system == "router_direct":
        decision = route_intent(prompt)
        hop = await caller(decision.model, prompt)
        execution = {
            "steps": [],
            "complete": bool(hop.get("ok")),
            "final_output": hop.get("content") or "",
            "final_step": "direct",
            "calls": 1,
            "latency_ms": hop.get("latency_ms") or 0.0,
            "gpu_seconds": hop.get("gpu_seconds") or 0.0,
            "direct": {
                **hop,
                "route": {
                    "model": decision.model,
                    "detected_intent": decision.intent,
                    "reason": decision.reason,
                },
            },
        }
    elif system == "oracle":
        plan = oracle
        execution = await execute_plan(plan, caller)
    elif system in PLANNER_PROFILES:
        planner_profile = system
        plan, planning = await generate_plan(prompt, caller, profile=system)
        execution = await execute_plan(plan, caller)
    else:
        raise ValueError(f"unknown system: {system}")

    planner_attempts = (planning or {}).get("attempts") or []
    planner_latency = sum(float(x.get("latency_ms") or 0) for x in planner_attempts)
    planner_gpu = sum(float(x.get("gpu_seconds") or 0) for x in planner_attempts)
    plan_match = bool(plan and plan_signature(plan) == plan_signature(oracle))
    expected_intents = [str(s["intent"]) for s in item["oracle_plan"]["steps"]]
    actual_intents = [str(s.intent) for s in plan.steps] if plan else []
    superfluous = max(0, len(actual_intents) - len(expected_intents))
    missing = max(0, len(expected_intents) - len(actual_intents))
    return {
        "id": item["id"],
        "bucket": "composite",
        "family": item["family"],
        "split": item.get("split"),
        "depth": item.get("depth"),
        "template_id_expected": item.get("template_id") or item.get("family"),
        "prompt": prompt,
        "provenance": item.get("provenance"),
        "system": system,
        "planner_profile": planner_profile,
        "oracle_plan": item["oracle_plan"],
        "final_step_expected": item["final_step"],
        "plan": plan.to_dict() if plan else None,
        "plan_valid": (
            (planning or {}).get("valid")
            if planning is not None
            else (True if plan is not None else None)
        ),
        "plan_match": plan_match,
        "workflow_signature": plan_signature(plan) if plan else None,
        "oracle_signature": plan_signature(oracle),
        "superfluous_steps": superfluous,
        "missing_steps": missing,
        "planning": planning,
        "selector_mode": (planning or {}).get("mode"),
        "selected_template": (planning or {}).get("template_id"),
        "fallback_reason": (planning or {}).get("fallback_reason"),
        "steps": execution["steps"],
        "direct": execution.get("direct"),
        "content": execution["final_output"],
        "http_status": 200 if execution["complete"] else 0,
        "complete": execution["complete"],
        "calls": int(execution["calls"]) + len(planner_attempts),
        "execution_calls": execution["calls"],
        "planner_calls": len(planner_attempts),
        "latency_ms": round(float(execution["latency_ms"]) + planner_latency, 1),
        "gpu_seconds": round(float(execution["gpu_seconds"]) + planner_gpu, 4),
        "decoding": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
    }


async def run(args: argparse.Namespace) -> None:
    bench = Path(args.bench)
    if not bench.is_absolute():
        bench = ROOT / bench
    items = filter_items(
        load_jsonl(bench),
        family=args.family,
        split=args.split,
        limit=args.limit,
    )
    if not items:
        raise SystemExit("no items left after filters")

    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    known = set(DIRECT_SYSTEMS) | ORCHESTRATED
    unknown = set(systems) - known
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    if not args.skip_health:
        await health_check(required_aliases(systems))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if "composite_v2" in bench.name:
        bench_tag = "v2"
    elif bench.name.startswith("agentcoma_uk_"):
        # e.g. agentcoma_uk_200_composite.jsonl -> agentcoma200
        bench_tag = "agentcoma" + bench.name.split("_")[2]
    else:
        bench_tag = "v1"
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for repeat in range(1, args.repeats + 1):
            for system in systems:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                name = (
                    f"{system}_composite_{bench_tag}_t{args.temperature:g}_"
                    f"s{args.seed}_rep{repeat}_{stamp}.jsonl"
                )
                path = out_dir / name
                rows: list[dict[str, Any]] = []
                with path.open("w", encoding="utf-8") as output:
                    for index, item in enumerate(items, 1):
                        row = await run_item(item, system, client, args)
                        row["repeat"] = repeat
                        rows.append(row)
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        print(
                            f"[{system} r{repeat} {index}/{len(items)}] {item['id']} "
                            f"ok={row['complete']} calls={row['calls']} "
                            f"{row['latency_ms']}ms"
                        )
                meta = {
                    "system": system,
                    "repeat": repeat,
                    "bench": str(bench),
                    "split_filter": args.split or None,
                    "n": len(rows),
                    "failures": sum(not row["complete"] for row in rows),
                    "calls": sum(row["calls"] for row in rows),
                    "latency_ms_total": round(sum(row["latency_ms"] for row in rows), 1),
                    "gpu_seconds_total": round(sum(row["gpu_seconds"] for row in rows), 4),
                    "decoding": {
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "max_tokens": args.max_tokens,
                    },
                }
                path.with_suffix(".meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        default="benchmarks/mixed_ua_composite_v1.jsonl",
    )
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--family", default="")
    parser.add_argument(
        "--split",
        default="",
        help="Comma-separated splits for v2 (train,dev,test). Empty = all.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--out-dir", default="results/week_10_composite")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
