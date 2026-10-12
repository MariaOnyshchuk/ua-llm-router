#!/usr/bin/env python3
"""How often does each routing method name the right intent and pick the right model?

Methods: rules v2; a small model as classifier (router/llm_intent.py); hybrid (rules
first, the model only when no rule matched). Two sets: hand-written everyday requests
(benchmarks/intent_natural_v1.jsonl) and a sample of mixed_ua_v7 dev in benchmark
wording. Needs the classifier model on its usual port. Wilson 95% intervals.

  PYTHONPATH=. python scripts/analysis/intent_natural_eval.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router.intent_rules import route_intent  # noqa: E402
from router.llm_intent import INTENT_MODEL, classify  # noqa: E402


def rows(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", default="mamay4")
    p.add_argument("--per-bucket", type=int, default=50)
    p.add_argument("--out", default="results/analysis/intent_natural_eval.json")
    args = p.parse_args()

    natural = [{"id": r["id"], "gold": r["intent"], "prompt": r["prompt"]}
               for r in rows(ROOT / "benchmarks/intent_natural_v1.jsonl")]
    dev = [r for r in rows(ROOT / "benchmarks/mixed_ua_v7.jsonl") if r.get("split") == "dev"]   # dev only
    rng = random.Random(42)
    bench = []
    for bucket in ("knowledge", "translate", "alignment", "instruct", "chat", "code"):
        pool = sorted((r for r in dev if r["bucket"] == bucket), key=lambda r: r["id"])
        for r in rng.sample(pool, min(args.per_bucket, len(pool))):
            bench.append({"id": r["id"], "gold": r["bucket"], "prompt": r["prompt"]})

    out = {"classifier": args.classifier, "sets": {}}
    with httpx.Client(timeout=120) as client:
        for name, items in (("natural", natural), ("v7_dev_sample", bench)):
            per = defaultdict(lambda: defaultdict(list))
            lat, unusable, llm_calls_hybrid, items_out = [], 0, 0, []
            for it in items:
                rule = route_intent(it["prompt"], profile="v2")
                matched = not rule.reason.startswith("default")
                res = classify(client, it["prompt"], args.classifier)
                lat.append(res["latency_ms"])
                llm = res["intent"]
                unusable += llm is None
                llm = llm or rule.intent                       # unusable reply: fall back to the rules
                hybrid = rule.intent if matched else llm
                llm_calls_hybrid += not matched
                gold = it["gold"]
                for method, intent in (("rules_v2", rule.intent), ("llm", llm), ("hybrid", hybrid)):
                    per[method]["intent"].append(intent == gold)
                    per[method]["model"].append(INTENT_MODEL.get(intent) == INTENT_MODEL[gold])
                    per[method][f"intent:{gold}"].append(intent == gold)
                items_out.append({"id": it["id"], "gold": gold, "rules_v2": rule.intent, "llm": llm, "hybrid": hybrid})
            n = len(items)
            summary = {}
            for method, d in per.items():
                summary[method] = {
                    "intent_accuracy": round(st.mean(d["intent"]), 3), "intent_ci95": wilson(sum(d["intent"]), n),
                    "model_accuracy": round(st.mean(d["model"]), 3), "model_ci95": wilson(sum(d["model"]), n),
                    "by_intent": {k.split(":")[1]: f"{sum(v)}/{len(v)}" for k, v in sorted(d.items()) if k.startswith("intent:")},
                }
            out["sets"][name] = {"n": n, "methods": summary, "llm_unusable_replies": unusable,
                                 "llm_latency_p50_ms": round(sorted(lat)[len(lat) // 2], 1),
                                 "hybrid_share_of_requests_sent_to_llm": round(llm_calls_hybrid / n, 3), "items": items_out}
            print(f"\n== {name} (n={n}); classifier p50 {out['sets'][name]['llm_latency_p50_ms']} ms; "
                  f"unusable replies {unusable}; hybrid asks the model on {llm_calls_hybrid}/{n}")
            print("| method | right intent | 95% | right model | 95% |\n|---|---:|---|---:|---|")
            for method, s in summary.items():
                print(f"| {method} | {s['intent_accuracy']:.2f} | {s['intent_ci95']} | {s['model_accuracy']:.2f} | {s['model_ci95']} |")
            for method, s in summary.items():
                print(f"  {method:8} by intent: {s['by_intent']}")
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
