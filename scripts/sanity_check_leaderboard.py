#!/usr/bin/env python3
"""Sanity-check our scoring pipeline against Ukrainian LLM leaderboard ballpark numbers.

This does NOT reproduce the full leaderboard harness (lm_eval). It runs a small
proxy set so you can catch broken scorers / broken decoding before trusting
diploma tables.

Usage (on lab, with mamay12 on :8002):
  PYTHONPATH=. python scripts/sanity_check_leaderboard.py --backend mamay12

Expected ballpark (from lang-uk Ukrainian LLM leaderboard, 0-shot IT models):
  Mamay-12B-IT-v2 IFEval UA ≈ 61.92
  Mamay-12B-IT-v2 MMLU UA   ≈ 64.26
  Mamay-12B-IT-v2 FLORES UA ≈ 35.24  (their metric scale ≠ our chrF — compare trend)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

BACKENDS = {
    "mamay12": (
        "http://127.0.0.1:8002/v1",
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
    ),
    "mamay4": (
        "http://127.0.0.1:8003/v1",
        "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
    ),
}

# Tiny IFEval-style instruction constraints (proxy, not official IFEval).
IFEVAL_PROXY = [
    {
        "id": "ifeval-proxy-001",
        "prompt": "Відповідай ЛИШЕ одним словом українською: так або ні. Чи 2+2=4?",
        "check": "yes_no",
    },
    {
        "id": "ifeval-proxy-002",
        "prompt": "Поверни ТІЛЬКИ валідний JSON об'єкт з ключами name (рядок) і score (число).",
        "check": "json_obj",
    },
    {
        "id": "ifeval-proxy-003",
        "prompt": "Напиши рівно 3 пункти списку Markdown, кожен рядок починається з '- '.",
        "check": "bullets3",
    },
]


def chat(base: str, model: str, prompt: str) -> str:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{base}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.0,
                "seed": 42,
            },
        )
        resp.raise_for_status()
        return ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def check(kind: str, text: str) -> bool:
    import json as _json
    import re

    t = (text or "").strip()
    if kind == "yes_no":
        return t.lower() in {"так", "ні", "yes", "no"}
    if kind == "json_obj":
        try:
            m = re.search(r"\{[\s\S]*\}", t)
            obj = _json.loads(m.group(0) if m else t)
            return isinstance(obj, dict) and "name" in obj and "score" in obj
        except Exception:
            return False
    if kind == "bullets3":
        lines = [ln for ln in t.splitlines() if ln.strip().startswith("- ")]
        return len(lines) == 3
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mamay12", choices=sorted(BACKENDS))
    p.add_argument("--out", type=Path, default=Path("results/sanity_leaderboard_proxy.json"))
    args = p.parse_args()

    base, model = BACKENDS[args.backend]
    # loud health check
    r = httpx.get(f"{base}/models", timeout=10.0)
    if r.status_code != 200:
        raise SystemExit(f"health check failed: {base}/models → {r.status_code}")

    rows = []
    for item in IFEVAL_PROXY:
        t0 = time.perf_counter()
        content = chat(base, model, item["prompt"])
        ms = (time.perf_counter() - t0) * 1000
        ok = check(item["check"], content)
        rows.append(
            {
                "id": item["id"],
                "ok": ok,
                "latency_ms": round(ms, 1),
                "content": content[:300],
            }
        )
        print(f"{item['id']}: {'PASS' if ok else 'FAIL'} {ms:.0f}ms | {content[:60]!r}")

    acc = sum(1 for r in rows if r["ok"]) / len(rows)
    out = {
        "backend": args.backend,
        "model": model,
        "proxy_ifeval_accuracy": round(acc, 4),
        "n": len(rows),
        "note": (
            "Proxy only — not official IFEval/MMLU. "
            "Use to catch broken serving/scoring before diploma tables."
        ),
        "leaderboard_ballpark": {
            "mamay12_ifeval_ua": 61.92,
            "mamay12_mmlu_ua": 64.26,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "proxy_ifeval_accuracy": acc}, ensure_ascii=False))


if __name__ == "__main__":
    main()
