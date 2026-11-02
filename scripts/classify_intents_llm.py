#!/usr/bin/env python3
"""Mamay-4B names the intent of every item (routing profile `llm`, router/llm_intent.py), with the latency
of each call. With the stored single-model answers this gives the score and latency of the `llm` router
(classifier call + the chosen model's answer), see scripts/analysis/plot_latency_quality.py.

Writes ids, intents and latencies only (no model answers):
  results/<bench>_lab/intent_llm_<alias>.jsonl

  python scripts/classify_intents_llm.py --bench graphs_mixed_100_dev --out-dir results/graphs_mixed_100_dev_lab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.llm_intent import classify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alias", default="mamay4")
    args = ap.parse_args()
    items = [json.loads(x) for x in (ROOT / "benchmarks" / f"{args.bench}.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip() and not x.startswith("#")]
    out = Path(args.out_dir) / f"intent_llm_{args.alias}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with httpx.Client(timeout=60.0) as client, out.open("w", encoding="utf-8") as f:
        for it in items:
            try:
                r = classify(client, it["prompt"], args.alias)
                row = {"id": it["id"], "intent": r["intent"], "reply": r["reply"][:40],
                       "latency_ms": round(r["latency_ms"], 1), "ok": True}
            except Exception as exc:  # noqa: BLE001
                failed += 1
                row = {"id": it["id"], "intent": None, "reply": "", "latency_ms": None, "ok": False,
                       "error": type(exc).__name__}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"wrote": str(out), "n": len(items), "failed": failed}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
