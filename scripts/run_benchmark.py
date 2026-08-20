#!/usr/bin/env python3
"""Run mixed_ua benchmark against router (auto) and/or explicit models.

Examples:
  PYTHONPATH=. python scripts/run_benchmark.py \\
    --input benchmarks/mixed_ua_v0.jsonl \\
    --system router --base-url http://127.0.0.1:4010/v1 --model auto

  PYTHONPATH=. python scripts/run_benchmark.py \\
    --input benchmarks/mixed_ua_v0.jsonl \\
    --system lapa --base-url http://127.0.0.1:4010/v1 --model lapa

  # Big-model baseline (OpenAI-compatible API):
  PYTHONPATH=. python scripts/run_benchmark.py \\
    --input benchmarks/mixed_ua_v0.jsonl \\
    --system big \\
    --base-url https://api.openai.com/v1 \\
    --model gpt-4.1-mini \\
    --api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def chat_completion(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    seed: int | None = 42,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    t0 = time.perf_counter()
    resp = client.post(url, headers=headers, json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000
    body: dict[str, Any]
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    content = ""
    if resp.is_success and isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
    return {
        "http_status": resp.status_code,
        "latency_ms": round(latency_ms, 1),
        "content": content,
        "router": body.get("router") if isinstance(body, dict) else None,
        "usage": body.get("usage") if isinstance(body, dict) else None,
        "error": None if resp.is_success else body,
        "decoding": {
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run mixed_ua benchmark")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--system", required=True, help="Label for this run, e.g. router|lapa|big")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base, .../v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42, help="vLLM/OpenAI seed for reproducibility")
    p.add_argument("--limit", type=int, default=0, help="If >0, only first N items")
    p.add_argument("--bucket", default="", help="Optional filter: chat|translate|instruct|knowledge|code")
    args = p.parse_args()

    items = load_jsonl(args.input)
    if args.bucket:
        items = [x for x in items if x.get("bucket") == args.bucket]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"{args.system}_{stamp}.jsonl"

    meta = {
        "system": args.system,
        "base_url": args.base_url,
        "model": args.model,
        "input": str(args.input),
        "n": len(items),
        "started_at": stamp,
        "decoding": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
    }
    print(json.dumps({"run": meta}, ensure_ascii=False))

    with httpx.Client(timeout=180.0) as client, out_path.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(items, 1):
            result = chat_completion(
                client,
                base_url=args.base_url,
                model=args.model,
                prompt=item["prompt"],
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            row = {
                "id": item.get("id"),
                "bucket": item.get("bucket"),
                "prompt": item.get("prompt"),
                "reference": item.get("reference", ""),
                "system": args.system,
                "model_requested": args.model,
                **result,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            preview = (result.get("content") or "")[:80].replace("\n", " ")
            print(f"[{i}/{len(items)}] {item.get('id')} {result['http_status']} {result['latency_ms']}ms | {preview}")

    print(json.dumps({"wrote": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
