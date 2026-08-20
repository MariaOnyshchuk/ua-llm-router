#!/usr/bin/env python3
"""Alignment ablation: Lapa vs Mamay-4B × baseline / clarified / few-shot.

Runs only the alignment slice of mixed_ua_v4_balanced (≈32 items).
Social items (0/1/2) get optional prompt variants; ethics left as-is.

Example (on lab, with :8001 and :8003 up):
  python scripts/run_alignment_ablation.py \\
    --bench benchmarks/mixed_ua_v4_balanced.jsonl \\
    --systems lapa:baseline,lapa:fewshot,mamay4:baseline,mamay4:fewshot \\
    --out-dir results/week_5_alignment_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.alignment_prompt_variants import apply_alignment_variant  # noqa: E402
from scripts.run_small_router_cluster import (  # noqa: E402
    BACKENDS,
    MAX_TOKENS,
    SEED,
    TEMPERATURE,
    chat,
    health_check,
    load_jsonl,
)
from scripts.score_results import score_row  # noqa: E402


def parse_systems(spec: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            model, variant = part.split(":", 1)
        else:
            model, variant = part, "baseline"
        out.append((model.strip(), variant.strip()))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", type=Path, default=ROOT / "benchmarks/mixed_ua_v4_balanced.jsonl")
    p.add_argument(
        "--systems",
        default="lapa:baseline,lapa:clarified,lapa:fewshot,mamay4:baseline,mamay4:fewshot",
        help="Comma list of model:variant",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/week_5_alignment_ablation")
    p.add_argument("--bucket", default="alignment")
    args = p.parse_args()

    items = [r for r in load_jsonl(args.bench) if r.get("bucket") == args.bucket]
    if not items:
        raise SystemExit(f"no {args.bucket} items in {args.bench}")

    systems = parse_systems(args.systems)
    required = sorted({m for m, _ in systems})
    health_check(required)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary: dict = {"stamp": stamp, "n_items": len(items), "systems": {}}

    with httpx.Client(timeout=180.0) as client:
        for model, variant in systems:
            if model not in BACKENDS:
                raise SystemExit(f"unknown model {model}")
            base, backend_model = BACKENDS[model]
            name = f"{model}_{variant}"
            tagged = [apply_alignment_variant(it, variant) for it in items]
            out = args.out_dir / f"{name}_align_v4_{stamp}.jsonl"
            scores_path = args.out_dir / f"{name}_align_v4_{stamp}.scores.jsonl"

            print(f"=== {name} n={len(tagged)} → {out.name}", flush=True)
            rows = []
            with out.open("w", encoding="utf-8") as fout:
                for i, item in enumerate(tagged, 1):
                    result = chat(client, base, backend_model, item["prompt"])
                    row = {
                        "id": item.get("id"),
                        "bucket": item.get("bucket"),
                        "prompt": item.get("prompt"),
                        "reference": item.get("reference", ""),
                        "notes": item.get("notes", ""),
                        "system": name,
                        "model_requested": model,
                        "prompt_variant": variant,
                        "http_status": result["http_status"],
                        "latency_ms": result["latency_ms"],
                        "gpu_seconds": result["gpu_seconds"],
                        "content": result["content"],
                        "usage": result["usage"],
                        "error": result["error"],
                        "decoding": result["decoding"],
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                    preview = (result.get("content") or "").replace("\n", " ")[:60]
                    print(
                        f"  [{i}/{len(tagged)}] {item.get('id')} ref={item.get('reference')} "
                        f"→ {preview!r} ({result['latency_ms']}ms)",
                        flush=True,
                    )
                    time.sleep(0)  # yield

            # score
            by_ref = {"0": [], "1": [], "2": []}
            social_ref1 = []
            scored_rows = []
            with scores_path.open("w", encoding="utf-8") as sf:
                for row in rows:
                    s = score_row(row)
                    rec = {
                        "id": row["id"],
                        "reference": row.get("reference"),
                        "score": s["score"],
                        "detail": s.get("detail"),
                        "content": (row.get("content") or "")[:200],
                        "is_social": str(row.get("id", "")).startswith("ualign-social"),
                    }
                    sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    scored_rows.append(rec)
                    ref = str(row.get("reference") or "")
                    if ref in by_ref:
                        by_ref[ref].append(float(s["score"]))
                    if rec["is_social"] and ref == "1":
                        social_ref1.append(rec)

            mean = sum(float(r["score"]) for r in scored_rows) / max(1, len(scored_rows))
            ref1_mean = (
                sum(by_ref["1"]) / len(by_ref["1"]) if by_ref["1"] else None
            )
            social_ref1_acc = (
                sum(1 for r in social_ref1 if r["score"] == 1.0) / len(social_ref1)
                if social_ref1
                else None
            )
            # collapse diagnostic: among social ref=1, how often model said 2
            said_2 = 0
            for r in social_ref1:
                m = __import__("re").search(r"(?<!\d)([0-2])(?!\d)", r.get("content") or "")
                if m and m.group(1) == "2":
                    said_2 += 1

            summary["systems"][name] = {
                "model": model,
                "variant": variant,
                "n": len(scored_rows),
                "overall_mean": round(mean, 4),
                "ref1_mean": None if ref1_mean is None else round(ref1_mean, 4),
                "social_ref1_n": len(social_ref1),
                "social_ref1_acc": None
                if social_ref1_acc is None
                else round(social_ref1_acc, 4),
                "social_ref1_answered_2": said_2,
                "per_ref_n": {k: len(v) for k, v in by_ref.items()},
                "per_ref_mean": {
                    k: (round(sum(v) / len(v), 4) if v else None) for k, v in by_ref.items()
                },
                "jsonl": str(out),
                "scores": str(scores_path),
            }
            print(
                f"  → mean={mean:.3f} social_ref1_acc={social_ref1_acc} "
                f"ref1→answered_2={said_2}/{len(social_ref1)}",
                flush=True,
            )

    summary_path = args.out_dir / f"summary_align_ablation_{stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
