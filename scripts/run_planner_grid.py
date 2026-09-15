#!/usr/bin/env python3
"""A2: orchestrator prompt × model grid — planning-only screening.

Runs ONLY the planning step (short JSON, no specialist execution) across
(planner_model, prompt_variant) and scores plan_valid + intent-sequence
match against oracle_plan on mixed_ua_composite_v1.

  PYTHONPATH=. python scripts/run_planner_grid.py \
      --bench benchmarks/mixed_ua_composite_v1.jsonl \
      --out results/tables/a2_planner_grid.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.backends import BACKENDS  # noqa: E402
from router.client import async_chat  # noqa: E402
from router.planner import (  # noqa: E402
    SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION_V3_MINIMAL,
    extract_json_object,
    generate_plan,
)

PROMPT_VARIANTS: dict[str, str] = {
    "current_v2": SYSTEM_INSTRUCTION,
    "minimal_v3": SYSTEM_INSTRUCTION_V3_MINIMAL,
}

DEFAULT_MODELS = ("mamay4", "mamay12")


@dataclass
class GridResult:
    model: str
    prompt_variant: str
    id: str
    family: str
    plan_valid: bool
    plan_match: bool
    raw_json_valid: bool
    raw_n_steps: int
    raw_intents: str
    finish_reason: str
    completion_tokens: int
    multi_step: bool
    has_translate: bool
    last_numeric: bool
    repaired: bool
    n_steps: int
    oracle_n_steps: int
    intents_predicted: str
    intents_oracle: str
    latency_ms: float


def load_items(path: Path | str) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        row = json.loads(line)
        rid = row.get("id")
        if rid and row.get("oracle_plan") and rid not in items:
            items[str(rid)] = {
                "prompt": row.get("prompt"),
                "oracle_plan": row["oracle_plan"],
                "family": row.get("family") or "",
            }
    return items


def intents_sequence(plan_like: Any) -> list[str]:
    if hasattr(plan_like, "steps"):
        return [s.intent for s in plan_like.steps]
    if isinstance(plan_like, dict) and "steps" in plan_like:
        return [str(s.get("intent") or "") for s in plan_like["steps"]]
    if isinstance(plan_like, list):
        return [str(s.get("intent") or "") for s in plan_like]
    return []


async def health_check(aliases: list[str]) -> None:
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for alias in aliases:
            base, _ = BACKENDS[alias]
            try:
                response = await client.get(f"{base}/models")
                if response.status_code != 200:
                    errors.append(f"{alias}: HTTP {response.status_code}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{alias}: {type(exc).__name__}: {exc}")
    if errors:
        raise SystemExit("Planner-grid preflight failed:\n  - " + "\n  - ".join(errors))


async def run_grid(args: argparse.Namespace) -> None:
    bench = Path(args.bench)
    if not bench.is_absolute():
        bench = ROOT / bench
    items = load_items(bench)
    if args.family:
        allowed = {x.strip() for x in args.family.split(",") if x.strip()}
        items = {rid: item for rid, item in items.items() if item.get("family") in allowed}
    if args.limit:
        items = dict(list(items.items())[: args.limit])
    if not items:
        raise SystemExit(f"no composite items with oracle_plan in {bench}")
    print(f"loaded {len(items)} composite items from {bench}")

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown_models = [m for m in models if m not in BACKENDS]
    if unknown_models:
        raise SystemExit(f"unknown models: {unknown_models}")
    unknown_variants = [v for v in variants if v not in PROMPT_VARIANTS]
    if unknown_variants:
        raise SystemExit(f"unknown variants: {unknown_variants}")
    if not args.skip_health:
        await health_check(models)

    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    plans_dir = None
    if args.save_plans:
        plans_dir = Path(args.save_plans)
        if not plans_dir.is_absolute():
            plans_dir = ROOT / plans_dir
        plans_dir.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(args.timeout)
    results: list[GridResult] = []
    async with httpx.AsyncClient(timeout=timeout) as client:

        async def caller(alias: str, prompt_text: str) -> dict[str, Any]:
            return await async_chat(
                client,
                alias,
                prompt_text,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )

        for variant_name in variants:
            instruction = PROMPT_VARIANTS[variant_name]
            for model in models:
                print(f"=== {model} / {variant_name} ===")
                if plans_dir is not None:
                    (plans_dir / f"{model}_{variant_name}.jsonl").unlink(missing_ok=True)
                for index, (rid, item) in enumerate(items.items(), 1):
                    plan, meta = await generate_plan(
                        str(item["prompt"]),
                        caller,
                        repair=True,
                        instruction=instruction,
                        model=model,
                    )
                    predicted = intents_sequence(plan)
                    oracle = intents_sequence(item["oracle_plan"])
                    attempts = meta.get("attempts") or []
                    latency = sum(float(x.get("latency_ms") or 0) for x in attempts)
                    final_attempt = attempts[-1] if attempts else {}
                    raw_json_valid = False
                    raw_intents: list[str] = []
                    try:
                        raw_plan = extract_json_object(final_attempt.get("content") or "")
                        raw_json_valid = True
                        raw_intents = intents_sequence(raw_plan)
                    except (ValueError, json.JSONDecodeError):
                        pass
                    row = GridResult(
                        model=model,
                        prompt_variant=variant_name,
                        id=rid,
                        family=str(item.get("family") or ""),
                        plan_valid=bool(meta.get("valid")),
                        plan_match=predicted == oracle,
                        raw_json_valid=raw_json_valid,
                        raw_n_steps=len(raw_intents),
                        raw_intents=">".join(raw_intents),
                        finish_reason=str(final_attempt.get("finish_reason") or ""),
                        completion_tokens=int(
                            (final_attempt.get("usage") or {}).get("completion_tokens") or 0
                        ),
                        multi_step=len(predicted) >= 2,
                        has_translate="translate" in predicted,
                        last_numeric=bool(predicted) and predicted[-1] in {"instruct", "code"},
                        repaired=bool(meta.get("repaired")),
                        n_steps=len(predicted),
                        oracle_n_steps=len(oracle),
                        intents_predicted=">".join(predicted),
                        intents_oracle=">".join(oracle),
                        latency_ms=round(latency, 1),
                    )
                    results.append(row)
                    print(
                        f"[{model} {variant_name} {index}/{len(items)}] {rid} "
                        f"valid={row.plan_valid} match={row.plan_match} "
                        f"{row.intents_predicted} {row.latency_ms}ms"
                    )
                    if plans_dir is not None:
                        payload = {
                            **asdict(row),
                            "input_prompt": str(item["prompt"]),
                            "plan": plan.to_dict(),
                            # Keep raw model responses for qualitative screening.
                            # Invalid JSON must remain inspectable rather than being
                            # hidden behind the single-hop fallback plan.
                            "planning": meta,
                        }
                        path = plans_dir / f"{model}_{variant_name}.jsonl"
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    fieldnames = list(asdict(results[0]).keys()) if results else []
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "n": 0,
            "valid": 0,
            "match": 0,
            "multi": 0,
            "translate": 0,
            "numeric": 0,
            "repaired": 0,
        }
    )
    print(
        f"\n{'model':10} {'prompt':12} {'n':>4} {'valid':>8} {'match':>8} "
        f"{'multi':>8} {'xlat':>8} {'last#':>8} {'repair':>8}"
    )
    for row in results:
        key = (row.model, row.prompt_variant)
        agg[key]["n"] += 1
        agg[key]["valid"] += int(row.plan_valid)
        agg[key]["match"] += int(row.plan_match)
        agg[key]["multi"] += int(row.multi_step)
        agg[key]["translate"] += int(row.has_translate)
        agg[key]["numeric"] += int(row.last_numeric)
        agg[key]["repaired"] += int(row.repaired)
    for (model, variant), values in sorted(agg.items()):
        n = values["n"] or 1
        print(
            f"{model:10} {variant:12} {int(n):>4} "
            f"{values['valid'] / n:>8.3f} {values['match'] / n:>8.3f} "
            f"{values['multi'] / n:>8.3f} {values['translate'] / n:>8.3f} "
            f"{values['numeric'] / n:>8.3f} {values['repaired'] / n:>8.3f}"
        )
    print(f"wrote {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="benchmarks/mixed_ua_composite_v1.jsonl")
    parser.add_argument("--out", default="results/tables/a2_planner_grid.csv")
    parser.add_argument("--save-plans", default="results/a2_planner_grid")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--variants", default=",".join(PROMPT_VARIANTS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--family",
        default="",
        help="Comma-separated families to keep (e.g. travel_agent,science_experiments)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_grid(args))


if __name__ == "__main__":
    main()
