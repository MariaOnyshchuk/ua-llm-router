#!/usr/bin/env python3
"""Frontier-API baseline on the same suite, scored by the same pipeline.

This is the *unavailable alternative* in the thesis framing: what an
organisation would get if it could send Ukrainian data to a hosted frontier
model. Runs from the laptop — no GPU, no cluster.

Row shape matches scripts/run_small_router_cluster.py so score_results.py and
aggregate_repeat_scores.py work unchanged.

  export OPENAI_API_KEY=sk-...
  python scripts/run_api_baseline.py \
    --system gpt --model gpt-4o-2024-11-20 \
    --align-prompt fewshot --repeats 3 \
    --out-dir results/week_9_baselines

Latency here is **network wall-clock**, not GPU time. Keep --concurrency 1 if
you intend to compare latency against the self-hosted systems.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.alignment_prompt_variants import apply_alignment_variant  # noqa: E402

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def chat_once(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    max_retries: int,
) -> dict[str, Any]:
    """One completion with backoff. Returns the runner's result dict shape."""
    last: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = client.post(url, json=payload, headers=headers)
        except Exception as exc:  # noqa: BLE001 — network errors are expected here
            ms = (time.perf_counter() - t0) * 1000
            last = {
                "http_status": 0,
                "latency_ms": round(ms, 1),
                "content": "",
                "usage": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return last

        ms = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}

        if resp.is_success:
            choice = (body.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            return {
                # Normalise to 200: score_results.py treats anything else as a miss.
                "http_status": 200,
                "latency_ms": round(ms, 1),
                "content": content,
                "usage": body.get("usage"),
                "error": None,
                "finish_reason": choice.get("finish_reason"),
            }

        last = {
            "http_status": resp.status_code,
            "latency_ms": round(ms, 1),
            "content": "",
            "usage": None,
            "error": body,
        }
        if resp.status_code in RETRY_STATUS and attempt < max_retries:
            retry_after = resp.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (attempt + 1)
            time.sleep(delay)
            continue
        return last
    return last


def run_once(
    items: list[dict],
    args: argparse.Namespace,
    bench_tag: str,
    rep: int,
) -> tuple[Path, dict[str, Any]]:
    dest = Path(args.out_dir) if args.out_dir else (ROOT / "results")
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_rep{rep}" if args.repeats > 1 else ""
    out = dest / f"{args.system}_{bench_tag}{suffix}_{stamp}.jsonl"

    key = os.getenv(args.api_key_env, "")
    if not key:
        raise SystemExit(f"Missing API key: export {args.api_key_env}=...")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{args.base_url.rstrip('/')}/chat/completions"

    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    failures = 0

    with httpx.Client(timeout=args.timeout) as client, out.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(items, 1):
            tagged = apply_alignment_variant(item, args.align_prompt)
            prompt = tagged["prompt"]
            payload: dict[str, Any] = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            if args.seed is not None:
                payload["seed"] = args.seed

            result = chat_once(client, url, headers, payload, max_retries=args.max_retries)
            latencies.append(float(result["latency_ms"]))
            usage = result.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            if result["http_status"] != 200:
                failures += 1

            row = {
                "id": tagged.get("id"),
                "bucket": tagged.get("bucket"),
                "prompt": prompt,
                "reference": tagged.get("reference", ""),
                "notes": tagged.get("notes", ""),
                "he_prompt": tagged.get("he_prompt", ""),
                "he_test": tagged.get("he_test", ""),
                "he_entry_point": tagged.get("he_entry_point", ""),
                "system": args.system,
                "prompt_variant": args.align_prompt,
                "model_requested": args.model,
                "router": {
                    "model": args.system,
                    "intent": "explicit",
                    "reason": "hosted API baseline (no routing)",
                    "backend_model": args.model,
                    "api_base": args.base_url,
                },
                "cascade": None,
                "ensemble": None,
                "decoding": {
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "max_tokens": args.max_tokens,
                },
                **result,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            preview = (result.get("content") or "")[:70].replace("\n", " ")
            print(
                f"[{i}/{len(items)}] {item['id']} → {args.model} "
                f"{result['http_status']} {result['latency_ms']}ms | {preview}"
            )

    cost = (
        prompt_tokens / 1_000_000 * args.price_in
        + completion_tokens / 1_000_000 * args.price_out
    )
    meta = {
        "system": args.system,
        "bench_tag": bench_tag,
        "repeat": rep,
        "n": len(items),
        "wrote": str(out),
        "prompt_variant": args.align_prompt,
        "provider": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": 1,
        },
        "decoding": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
        "http_failures": failures,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "cost_usd_estimate": round(cost, 4),
        "price_per_1m": {"input": args.price_in, "output": args.price_out},
        "resource_protocol": {
            "note": (
                "Hosted API: latency is network wall-clock (client-side), not GPU time. "
                "No VRAM figure exists for this system — do not put it in a GPU-seconds "
                "or peak-VRAM column. Requests are sequential so p50 is comparable to "
                "the sequential self-hosted runs."
            ),
        },
        "latency_avg_ms": round(sum(latencies) / max(1, len(latencies)), 1),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))
    return out, meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hosted frontier-API baseline on mixed_ua_*, same protocol as the cluster runs."
    )
    p.add_argument("--system", default="api", help="System name used by score_results.py")
    p.add_argument("--model", required=True, help="Provider model id, e.g. gpt-4o-2024-11-20")
    p.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible base URL (OpenRouter / Azure / local gateway all work)",
    )
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--bench", default="benchmarks/mixed_ua_v4_balanced.jsonl")
    p.add_argument("--bucket", default="", help="Optional bucket filter, comma-separated")
    p.add_argument("--id-prefix", default="", help="Optional id prefix filter")
    p.add_argument("--limit", type=int, default=0, help="Only first N items (smoke test)")
    p.add_argument(
        "--align-prompt",
        default="fewshot",
        choices=["baseline", "fewshot", "clarified"],
        help="Match the router's alignment prompt (default fewshot, as in the product).",
    )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    p.add_argument("--out-dir", default="results/week_9_baselines")
    args = p.parse_args()

    bench_path = Path(args.bench)
    if not bench_path.is_absolute():
        bench_path = ROOT / bench_path
    items = load_jsonl(bench_path)
    if args.bucket:
        buckets = {b.strip() for b in args.bucket.split(",") if b.strip()}
        items = [x for x in items if x.get("bucket") in buckets]
    if args.id_prefix:
        items = [x for x in items if str(x.get("id", "")).startswith(args.id_prefix)]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No items after filters in {bench_path}")

    bench_tag = (
        f"{bench_path.stem}_t{args.temperature:g}_s{args.seed}"
        f"{'_' + args.align_prompt if args.align_prompt != 'baseline' else ''}"
    )

    total_cost = 0.0
    for rep in range(1, args.repeats + 1):
        _, meta = run_once(items, args, bench_tag, rep)
        total_cost += float(meta["cost_usd_estimate"])
    if args.repeats > 1:
        print(json.dumps({"repeats": args.repeats, "cost_usd_total": round(total_cost, 4)}))


if __name__ == "__main__":
    main()
