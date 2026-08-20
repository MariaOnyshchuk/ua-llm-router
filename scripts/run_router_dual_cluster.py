#!/usr/bin/env python3
"""Cluster-side dual-backend router bench: rules → :8001 Lapa / :8002 Mamay."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.intent_rules import route_intent  # noqa: E402

BACKENDS = {
    "lapa": ("http://127.0.0.1:8001/v1", "lapa-llm/lapa-v0.1.2-instruct"),
    "mamay": (
        "http://127.0.0.1:8002/v1",
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def main() -> None:
    items = load_jsonl(ROOT / "benchmarks" / "mixed_ua_v0.jsonl")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"router_dual_{stamp}.jsonl"

    with httpx.Client(timeout=180.0) as client, out.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(items, 1):
            decision = route_intent(item["prompt"])
            # map any leftover aliases to mamay
            alias = decision.model if decision.model in BACKENDS else "mamay"
            base, model = BACKENDS[alias]
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": item["prompt"]}],
                "max_tokens": 256,
                "temperature": 0.2,
            }
            t0 = time.perf_counter()
            resp = client.post(f"{base}/chat/completions", json=payload)
            ms = (time.perf_counter() - t0) * 1000
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
            content = ""
            if resp.is_success:
                content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            row = {
                "id": item.get("id"),
                "bucket": item.get("bucket"),
                "prompt": item.get("prompt"),
                "reference": item.get("reference", ""),
                "system": "router_dual",
                "model_requested": "auto",
                "http_status": resp.status_code,
                "latency_ms": round(ms, 1),
                "content": content,
                "router": {
                    "model": alias,
                    "intent": decision.intent,
                    "reason": decision.reason,
                    "backend_model": model,
                    "api_base": base,
                },
                "usage": body.get("usage") if isinstance(body, dict) else None,
                "error": None if resp.is_success else body,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i}/{len(items)}] {item['id']} → {alias} {resp.status_code} {ms:.0f}ms | {content[:70].replace(chr(10),' ')}")

    print(json.dumps({"wrote": str(out)}))


if __name__ == "__main__":
    main()
