#!/usr/bin/env python3
"""Run direct and orchestrated systems on composite benchmarks (v1/v2)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
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
from router.planner import GATED_PLANNERS, PLANNER_MODEL, PLANNER_PROFILES, generate_plan  # noqa: E402
from router.paper_planners import (DEMO_IDS, PAPER_PLANNER_MODELS, PAPER_PLANNERS, run_compiler, run_hgpt,
                                    run_planner_v2)  # noqa: E402

DIRECT_SYSTEMS = ("mamay4", "lapa", "aya", "qwen7", "mamay12", "mamay27")
DEFAULT_SYSTEMS = ("mamay4", "lapa", "aya", "qwen7", "router_direct", "oracle", "hybrid")
ORCHESTRATED = frozenset({"router_direct", "oracle", *PLANNER_PROFILES, *PAPER_PLANNERS})
# Learned (RouterBench-style) routing, router/artifacts/{knn,mlp}:
#   router_knn / router_mlp      one call, the learned router picks the model for the request;
#   <plan>+knn / <plan>+mlp      plan from `oracle` or a planner profile, every step routed by the
#                                learned router on the step's own prompt.
LEARNED_PROFILES = ("knn", "mlp", "knn_nodes", "mlp_nodes", "knn_nodes_only", "mlp_nodes_only", "mlp_deep_nodes_only")
LEARNED_DIRECT = {f"router_{k}": k for k in LEARNED_PROFILES}


def split_routing(system: str) -> tuple[str, str]:
    """'oracle+knn' -> ('oracle', 'knn'); plain names keep the rules (v2)."""
    base, _, routing = system.partition("+")
    return base, routing or "v2"


def known_system(system: str) -> bool:
    if system in DIRECT_SYSTEMS or system in ORCHESTRATED or system in LEARNED_DIRECT:
        return True
    base, routing = split_routing(system)
    return routing in LEARNED_PROFILES and (base == "oracle" or base in PLANNER_PROFILES or base in PAPER_PLANNERS)


def system_params(system: str) -> dict[str, str]:
    """Planner, planner model and step router of a system name, for MLflow run parameters."""
    base, routing = split_routing(system)
    if system in DIRECT_SYSTEMS:
        return {"planner": "none", "planner_model": "-", "router": "none (one model)"}
    if system == "router_direct" or system in LEARNED_DIRECT:
        return {"planner": "none", "planner_model": "-", "router": LEARNED_DIRECT.get(system, "v2")}
    if base == "oracle":
        return {"planner": "gold plan", "planner_model": "-", "router": routing}
    model = PAPER_PLANNER_MODELS.get(base) or GATED_PLANNERS.get(base) or PLANNER_MODEL
    if base.endswith("_mamay12"):
        model = "mamay12"
    return {"planner": base.replace(f"_{model}", ""), "planner_model": model, "router": routing}


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
        base, _ = split_routing(system)
        if system in LEARNED_DIRECT or base in {"router_direct", "oracle"} or base in PLANNER_PROFILES or base in PAPER_PLANNERS:
            required.update({"mamay4", "lapa", "aya"})
        if system in {"hybrid_mamay12", "hybrid_gated_mamay12"}:
            required.add("mamay12")
        if base == "hybrid_gated_mamay27" or PAPER_PLANNER_MODELS.get(base) == "mamay27":
            required.add("mamay27")
    return required


def as_plan_item(item: dict[str, Any]) -> dict[str, Any]:
    """A single-skill node item (benchmarks/mixed_ua_nodes_{train,test}.jsonl) becomes a one-step plan, so the
    same runner and scorer handle it. Its intent is the item's label (math items: `math` since 9 Nov 2026, `chat` before)."""
    if "oracle_plan" in item:
        return item
    step = {"id": "n1", "intent": item["label"], "depends_on": [], "prompt": item["prompt"],
            "rubric": item["check"]}
    return {**item, "family": f"node_{item['skill']}", "depth": 1, "final_step": "n1",
            "oracle_plan": {"steps": [step]}}


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
    elif system == "router_direct" or system in LEARNED_DIRECT:
        decision = route_intent(prompt, profile=LEARNED_DIRECT.get(system, "v2"))
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
    elif split_routing(system)[0] == "oracle":
        plan = oracle
        execution = await execute_plan(plan, caller, profile=split_routing(system)[1])
    elif split_routing(system)[0] in PAPER_PLANNERS:
        base, routing = split_routing(system)
        planner_profile = base
        model_kw = {"planner_model": PAPER_PLANNER_MODELS[base]} if PAPER_PLANNER_MODELS[base] else {}
        if base.startswith("planner_v2"):  # no demos
            out = await run_planner_v2(prompt, caller, profile=routing, **model_kw)
        else:
            run_paper = run_hgpt if base.startswith("hgpt") else run_compiler
            out = await run_paper(prompt, caller, profile=routing, demos=args.demo_items, **model_kw)
        plan, planning, execution = out["plan"], out["planning"], out["execution"]
    elif split_routing(system)[0] in PLANNER_PROFILES:
        base, routing = split_routing(system)
        planner_profile = base
        plan, planning = await generate_plan(prompt, caller, profile=base)
        execution = await execute_plan(plan, caller, profile=routing)
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
        [as_plan_item(item) for item in load_jsonl(bench)],
        family=args.family,
        split=args.split,
        limit=args.limit,
    )
    if not items:
        raise SystemExit("no items left after filters")
    # Demonstrations of the paper planners (router/paper_planners.py) come from graph-dev; with
    # --hold-out-demos those items are left out for every system, so all systems see the same items.
    by_id = {str(item["id"]): item for item in items}
    args.demo_items = [by_id[i] for i in DEMO_IDS if i in by_id]
    if args.hold_out_demos:
        items = [item for item in items if str(item["id"]) not in set(DEMO_IDS)]

    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    unknown = {s for s in systems if not known_system(s)}
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
    elif "composite_probe100" in bench.name:
        bench_tag = "probe100"
    elif bench.name.startswith(("graphs_mixed_", "nodes_mixed_")):
        # graphs_mixed_100_dev, graphs_mixed_100_test, nodes_mixed_600_train, nodes_mixed_600_test
        bench_tag = bench.stem
    elif bench.name.startswith("graphs_agentcoma_"):
        # e.g. graphs_agentcoma_200.jsonl -> agentcoma200
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
                tracker = None
                if args.mlflow:
                    from mlflow_tracking import RunLogger  # noqa: PLC0415

                    tracker = RunLogger(
                        Path(args.mlflow), args.mlflow_experiment or bench_tag, path.stem,
                        {**system_params(system), "system": system, "bench": bench.name,
                         "split": args.split or "all", "n": len(items), "repeat": repeat,
                         "temperature": args.temperature, "seed": args.seed, "max_tokens": args.max_tokens,
                         "hold_out_demos": args.hold_out_demos, "out_dir": str(out_dir.relative_to(ROOT))
                         if out_dir.is_relative_to(ROOT) else str(out_dir)},
                    )
                rows: list[dict[str, Any]] = []
                with path.open("w", encoding="utf-8") as output:
                    for index, item in enumerate(items, 1):
                        started_ns = time.time_ns()
                        row = await run_item(item, system, client, args)
                        row["repeat"] = repeat
                        if tracker:
                            tracker.item(row, started_ns, time.time_ns())
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
                if tracker:
                    tracker.finish({
                        "n": meta["n"], "failures": meta["failures"], "calls_per_item": meta["calls"] / max(1, meta["n"]),
                        "latency_avg_ms": meta["latency_ms_total"] / max(1, meta["n"]),
                        "gpu_seconds_per_item": meta["gpu_seconds_total"] / max(1, meta["n"]),
                    })
                print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        default="benchmarks/archive/mixed_ua_composite_v1.jsonl",
    )
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--family", default="")
    parser.add_argument(
        "--split",
        default="",
        help="Comma-separated splits for v2 (train,dev,test). Empty = all.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hold-out-demos", action="store_true",
                        help="leave out the demonstration items of the paper planners (router/paper_planners.DEMO_IDS)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--out-dir", default="results/week_10_composite")
    parser.add_argument("--mlflow", default="", metavar="DIR",
                        help="also log runs and per-item traces to a local MLflow store in DIR "
                             "(e.g. logs/<group>/mlflow; holds prompts and answers, stays out of git)")
    parser.add_argument("--mlflow-experiment", default="",
                        help="MLflow experiment name (default: the benchmark tag, e.g. graphs_mixed_100_dev)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
