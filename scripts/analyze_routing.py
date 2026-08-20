#!/usr/bin/env python3
"""Routing diagnostics: intent vs benchmark bucket, model oracle, code A/B."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.intent_rules import route_intent  # noqa: E402

# With current live backends, oracle model = qwen for code, mamay4 elsewhere.
ORACLE_MODEL = {
    "chat": "mamay4",
    "translate": "mamay4",
    "instruct": "mamay4",
    "knowledge": "mamay4",
    "code": "qwen",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def routing_report(bench_path: Path) -> dict:
    items = load_jsonl(bench_path)
    intent_hits = intent_misses = model_hits = model_misses = 0
    by_bucket: dict[str, dict[str, int]] = defaultdict(lambda: {"intent_ok": 0, "model_ok": 0, "n": 0})
    mismatches: list[dict] = []

    for item in items:
        bucket = item["bucket"]
        d = route_intent(item["prompt"])
        intent_ok = d.intent == bucket or (bucket == "chat" and d.intent == "chat")
        model_ok = d.model == ORACLE_MODEL[bucket]
        by_bucket[bucket]["n"] += 1
        if intent_ok:
            intent_hits += 1
            by_bucket[bucket]["intent_ok"] += 1
        else:
            intent_misses += 1
            mismatches.append(
                {
                    "id": item["id"],
                    "bucket": bucket,
                    "intent": d.intent,
                    "model": d.model,
                    "reason": d.reason,
                }
            )
        if model_ok:
            model_hits += 1
            by_bucket[bucket]["model_ok"] += 1
        else:
            model_misses += 1

    n = len(items)
    return {
        "n": n,
        "intent_accuracy": round(intent_hits / n, 3),
        "model_oracle_accuracy": round(model_hits / n, 3),
        "unique_backends_used": len({route_intent(i["prompt"]).model for i in items}),
        "by_bucket": {
            b: {
                **v,
                "intent_acc": round(v["intent_ok"] / v["n"], 3),
                "model_acc": round(v["model_ok"] / v["n"], 3),
            }
            for b, v in sorted(by_bucket.items())
        },
        "intent_mismatches": mismatches,
    }


def code_ab_from_detail(detail_path: Path) -> dict:
    rows = load_jsonl(detail_path)
    code = [r for r in rows if r.get("bucket") == "code"]
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in code:
        by_sys[r["system"]].append(r)

    def summarize(name: str) -> dict | None:
        xs = by_sys.get(name)
        if not xs:
            return None
        scores = [x["score"] for x in xs]
        lat = [x["latency_ms"] for x in xs]
        return {
            "n": len(xs),
            "mean_score": round(sum(scores) / len(scores), 4),
            "failures": [x["id"] for x in xs if x["score"] < 1.0],
            "latency_avg_ms": round(sum(lat) / len(lat), 1),
        }

    return {
        "mamay4": summarize("mamay4"),
        "qwen_via_router": summarize("router_small"),
        "mamay12": summarize("mamay12"),
    }


def main() -> None:
    bench = ROOT / "benchmarks" / "mixed_ua_v0.jsonl"
    detail = ROOT / "results" / "scores_small_vs_large_detail.jsonl"

    report = {"routing": routing_report(bench)}
    if detail.exists():
        report["code_bucket_ab"] = code_ab_from_detail(detail)

    out = ROOT / "results" / "routing_diagnostics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
