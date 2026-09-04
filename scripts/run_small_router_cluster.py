#!/usr/bin/env python3
"""Cluster-side small-specialist router bench: rules → Mamay4 :8003 / Qwen3B :8004.

Decoding is pinned for reproducibility:
  temperature=0, seed=42, max_tokens=256

Resource protocol (for diploma efficiency tables):
  Prefer one live model at a time. Record steady-state + peak VRAM via
  nvidia-smi sampling, and GPU-seconds ≈ wall latency (1 GPU assumed).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.backends import BACKENDS  # noqa: E402
from router.cascade import (  # noqa: E402
    CASCADE_BUCKETS,
    confidence_for_bucket,
    should_retry_mamay4_micro,
)
from router.client import sync_chat  # noqa: E402
from router.ensemble import (  # noqa: E402
    ENSEMBLE_VOTERS,
    extract_vote_label,
    majority_winner,
    should_ensemble_vote,
)
from router.intent_rules import BEST_BY_BUCKET, ROUTER_MODELS, route_intent  # noqa: E402
from scripts.alignment_prompt_variants import apply_alignment_variant  # noqa: E402

# Pinned decoding — do not change between systems without bumping run tags.
TEMPERATURE = 0.0
SEED = 42
MAX_TOKENS = 256


def health_check(required: list[str]) -> None:
    """Fail loud if a required backend is down (no silent fallbacks)."""
    errors = []
    with httpx.Client(timeout=5.0) as client:
        for alias in required:
            if alias not in BACKENDS:
                errors.append(f"{alias}: unknown backend")
                continue
            base, _ = BACKENDS[alias]
            url = f"{base}/models"
            try:
                r = client.get(url)
                if r.status_code != 200:
                    errors.append(f"{alias} {url} → HTTP {r.status_code}")
                else:
                    print(f"health ok: {alias} {url}")
            except Exception as e:
                errors.append(f"{alias} {url} → {type(e).__name__}: {e}")
    if errors:
        raise SystemExit(
            "Preflight health check FAILED (fallbacks disabled for eval):\n  - "
            + "\n  - ".join(errors)
        )

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def gpu_memory_snapshot() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as e:
        return [{"error": str(e)}]
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, used, total = parts[:4]
        util = float(parts[4]) if len(parts) > 4 and parts[4] not in ("", "[N/A]") else None
        rows.append(
            {
                "index": int(idx),
                "name": name,
                "memory_used_mb": float(used),
                "memory_total_mb": float(total),
                "utilization_gpu_pct": util,
            }
        )
    return rows


def summarize_vram(samples: list[list[dict]]) -> dict:
    """Aggregate nvidia-smi samples into peak / steady-state figures."""
    totals: list[float] = []
    peaks_single: list[float] = []
    for snap in samples:
        if not snap or "error" in snap[0]:
            continue
        used = [g["memory_used_mb"] for g in snap]
        totals.append(sum(used))
        peaks_single.append(max(used))
    if not totals:
        return {"n_samples": 0}
    # Steady-state ≈ median of total used; peak = max observed.
    totals_sorted = sorted(totals)
    mid = totals_sorted[len(totals_sorted) // 2]
    return {
        "n_samples": len(totals),
        "steady_total_vram_mb": round(mid, 1),
        "peak_total_vram_mb": round(max(totals), 1),
        "peak_single_gpu_mb": round(max(peaks_single), 1),
        "steady_total_vram_gb": round(mid / 1024.0, 3),
        "peak_total_vram_gb": round(max(totals) / 1024.0, 3),
        "peak_single_gpu_gb": round(max(peaks_single) / 1024.0, 3),
    }


class VramSampler:
    """Background nvidia-smi poller for peak / steady-state VRAM."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[list[dict]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = [gpu_memory_snapshot()]
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.samples.append(gpu_memory_snapshot())

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.samples.append(gpu_memory_snapshot())
        return summarize_vram(self.samples)


def chat(
    client: httpx.Client,
    base: str,
    model: str,
    prompt: str,
    *,
    temperature: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
) -> dict:
    # Resolve at call time so --seed-jitter can mutate module globals.
    temperature = TEMPERATURE if temperature is None else temperature
    seed = SEED if seed is None else seed
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    alias = next(
        (name for name, target in BACKENDS.items() if target == (base, model)),
        "",
    )
    result = sync_chat(
        client,
        alias,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    result["decoding"] = {
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    return result


def run_system(
    name: str,
    pick_backend,
    items: list[dict],
    bench_tag: str,
    *,
    vram_interval_s: float = 0.5,
    cascade: bool = False,
    cascade_kind: str = "legacy",
    ensemble: bool = False,
    align_prompt: str = "baseline",
    out_dir: Path | None = None,
) -> Path:
    dest = Path(out_dir) if out_dir else (ROOT / "results")
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = dest / f"{name}_{bench_tag}_{stamp}.jsonl"

    sampler = VramSampler(interval_s=vram_interval_s)
    sampler.start()
    vram_before = sampler.samples[0]
    latencies: list[float] = []
    escalate_count = 0
    ensemble_voted = 0
    ensemble_flipped = 0

    with httpx.Client(timeout=180.0) as client, out.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(items, 1):
            tagged = apply_alignment_variant(item, align_prompt)
            prompt = tagged["prompt"]
            alias, base, model, router_meta = pick_backend(tagged)
            result = chat(client, base, model, prompt)
            cascade_meta = None

            if cascade:
                if cascade_kind == "micro":
                    conf = should_retry_mamay4_micro(tagged, result.get("content") or "")
                    cascade_meta = {
                        "kind": "micro",
                        "first_model": alias,
                        "confidence": conf.confidence,
                        "confidence_reason": conf.reason,
                        "escalated": conf.escalate,
                    }
                    if conf.escalate:
                        escalate_count += 1
                        first = result
                        b4, m4 = BACKENDS["mamay4"]
                        result = chat(client, b4, m4, prompt)
                        result["latency_ms"] = round(
                            float(first["latency_ms"]) + float(result["latency_ms"]), 1
                        )
                        result["gpu_seconds"] = round(result["latency_ms"] / 1000.0, 4)
                        result["cascade_first_content"] = (first.get("content") or "")[:500]
                        router_meta = {
                            **router_meta,
                            "model": "mamay4",
                            "intent": router_meta.get("intent"),
                            "reason": f"micro cascade ← {conf.reason}",
                            "backend_model": m4,
                            "api_base": b4,
                        }
                        cascade_meta["final_model"] = "mamay4"
                    else:
                        cascade_meta["final_model"] = alias
                elif tagged.get("bucket") in CASCADE_BUCKETS:
                    conf = confidence_for_bucket(
                        str(tagged.get("bucket")), result.get("content") or ""
                    )
                    cascade_meta = {
                        "kind": "legacy",
                        "first_model": alias,
                        "confidence": conf.confidence,
                        "confidence_reason": conf.reason,
                        "escalated": conf.escalate,
                    }
                    if conf.escalate:
                        escalate_count += 1
                        base12, model12 = BACKENDS["mamay12"]
                        first = result
                        result = chat(client, base12, model12, prompt)
                        result["latency_ms"] = round(
                            float(first["latency_ms"]) + float(result["latency_ms"]), 1
                        )
                        result["gpu_seconds"] = round(result["latency_ms"] / 1000.0, 4)
                        result["cascade_first_content"] = (first.get("content") or "")[:500]
                        router_meta = {
                            **router_meta,
                            "model": "mamay12",
                            "intent": router_meta.get("intent"),
                            "reason": f"cascade escalate ← {conf.reason}",
                            "backend_model": model12,
                            "api_base": base12,
                        }
                        cascade_meta["final_model"] = "mamay12"
                    else:
                        cascade_meta["final_model"] = alias

            ensemble_meta = None
            if ensemble and should_ensemble_vote(tagged):
                ensemble_voted += 1
                first_alias = alias
                ballots: list[tuple[str, str | None, dict]] = [
                    (alias, extract_vote_label(tagged, result.get("content") or ""), result)
                ]
                extra_ms = 0.0
                for voter in ENSEMBLE_VOTERS:
                    if voter == first_alias:
                        continue
                    vb, vm = BACKENDS[voter]
                    extra = chat(client, vb, vm, prompt)
                    extra_ms += float(extra["latency_ms"])
                    ballots.append(
                        (voter, extract_vote_label(tagged, extra.get("content") or ""), extra)
                    )
                win_alias, win_lab, win_res = majority_winner(ballots, first_alias)
                summed = round(float(result["latency_ms"]) + extra_ms, 1)
                ensemble_meta = {
                    "kind": "majority_discrete",
                    "voted": True,
                    "first_model": first_alias,
                    "ballots": {a: lab for a, lab, _ in ballots},
                    "winner_label": win_lab,
                    "winner_model": win_alias,
                    "flipped": win_alias != first_alias,
                }
                if win_alias != first_alias:
                    ensemble_flipped += 1
                    result = dict(win_res)
                    router_meta = {
                        **router_meta,
                        "model": win_alias,
                        "intent": router_meta.get("intent"),
                        "reason": f"ensemble majority {win_lab} ← {first_alias}",
                        "backend_model": BACKENDS[win_alias][1],
                        "api_base": BACKENDS[win_alias][0],
                    }
                result["latency_ms"] = summed
                result["gpu_seconds"] = round(summed / 1000.0, 4)
                result["ensemble_first_content"] = (ballots[0][2].get("content") or "")[:500]
            elif ensemble:
                ensemble_meta = {"kind": "majority_discrete", "voted": False}

            latencies.append(float(result["latency_ms"]))
            row = {
                "id": tagged.get("id"),
                "bucket": tagged.get("bucket"),
                "prompt": prompt,
                "reference": tagged.get("reference", ""),
                "notes": tagged.get("notes", ""),
                "he_prompt": tagged.get("he_prompt", ""),
                "he_test": tagged.get("he_test", ""),
                "he_entry_point": tagged.get("he_entry_point", ""),
                "system": name,
                "prompt_variant": align_prompt,
                "model_requested": router_meta.get("model", alias),
                "router": router_meta,
                "cascade": cascade_meta,
                "ensemble": ensemble_meta,
                **result,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            preview = (result.get("content") or "")[:70].replace("\n", " ")
            tag = ""
            if cascade_meta and cascade_meta.get("escalated"):
                dest_model = cascade_meta.get("final_model") or "retry"
                tag = f" [ESCALATED→{dest_model}]"
            if ensemble_meta and ensemble_meta.get("voted"):
                tag += (
                    f" [VOTE {ensemble_meta.get('ballots')} → "
                    f"{ensemble_meta.get('winner_model')}:{ensemble_meta.get('winner_label')}]"
                )
            print(
                f"[{i}/{len(items)}] {item['id']} → {row['model_requested']} "
                f"{result['http_status']} {result['latency_ms']}ms{tag} | {preview}"
            )

    vram_stats = sampler.stop()
    gpu_seconds_total = sum(latencies) / 1000.0
    meta = {
        "system": name,
        "bench_tag": bench_tag,
        "n": len(items),
        "wrote": str(out),
        "prompt_variant": align_prompt,
        "decoding": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
        },
        "cascade": {
            "enabled": cascade,
            "kind": cascade_kind if cascade else None,
            "buckets": sorted(CASCADE_BUCKETS),
            "escalated": escalate_count,
            "escalate_rate": round(escalate_count / max(1, len(items)), 4),
        },
        "ensemble": {
            "enabled": ensemble,
            "voters": list(ENSEMBLE_VOTERS),
            "voted": ensemble_voted,
            "vote_rate": round(ensemble_voted / max(1, len(items)), 4),
            "flipped": ensemble_flipped,
            "flip_rate": round(ensemble_flipped / max(1, ensemble_voted), 4),
        },
        "resource_protocol": {
            "note": (
                "Prefer one model live at a time for fair weight-VRAM. "
                "With gpu_memory_utilization≈0.9, nvidia-smi shows KV-cache reservation, "
                "not bare weight footprint. gpu_seconds ≈ wall latency (1 GPU)."
            ),
            "vram_sample_interval_s": vram_interval_s,
        },
        "vram_before": vram_before,
        "vram_after": gpu_memory_snapshot(),
        "vram": vram_stats,
        "gpu_seconds_total": round(gpu_seconds_total, 3),
        "gpu_seconds_per_prompt": round(gpu_seconds_total / max(1, len(items)), 4),
        "latency_avg_ms": round(sum(latencies) / max(1, len(latencies)), 1),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Run on the GPU box (ucu-lab-2240). Needs vLLM backends. "
            "Fallbacks are disabled for eval — preflight health check fails loud."
        )
    )
    MODE_CHOICES = [
        "all",
        "mamay4",
        "mamay12",
        "qwen",
        "qwen7",
        "lapa",
        "aya",
        "router_matrix",
        "router_small",
        "router_cascade",
        "router_cascade_micro",
        "router_best",  # gold-bucket → v4 specialist winners (oracle, not rules)
        "router_ensemble",  # rules v2 + majority vote on alignment + ZNO
    ]
    p.add_argument("mode_pos", nargs="?", default=None, choices=MODE_CHOICES)
    p.add_argument(
        "--mode",
        dest="mode_flag",
        default=None,
        choices=MODE_CHOICES,
        help="Same as positional mode (plan-style --mode mamay4).",
    )
    p.add_argument(
        "--align-prompt",
        default="baseline",
        choices=["baseline", "fewshot", "clarified"],
        help="Rewrite social alignment prompts before chat (eval-only).",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Directory for JSONL + meta (default: results/).",
    )
    p.add_argument(
        "--bench",
        default="benchmarks/mixed_ua_v4_balanced.jsonl",
        help="JSONL suite (default: mixed_ua_v4_balanced)",
    )
    p.add_argument(
        "--bucket",
        default="",
        help="Optional filter, e.g. knowledge or code (comma-separated OK)",
    )
    p.add_argument(
        "--id-prefix",
        default="",
        help="Optional id prefix filter, e.g. zno- for ZNO-only",
    )
    p.add_argument(
        "--vram-interval",
        type=float,
        default=0.5,
        help="nvidia-smi sample interval seconds (default 0.5)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the whole system run N times (for mean±sd). Same seed unless --seed-jitter.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override decoding max_tokens (default 256). Use 512 for HumanEval completions.",
    )
    p.add_argument(
        "--seed-jitter",
        action="store_true",
        help="Use seed, seed+1, … across repeats (otherwise identical seed measures residual GPU noise).",
    )
    p.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip preflight /v1/models checks (not recommended for eval).",
    )
    args = p.parse_args()
    args.mode = args.mode_flag or args.mode_pos or "all"

    bench_path = Path(args.bench)
    if not bench_path.is_absolute():
        bench_path = ROOT / bench_path
    items = load_jsonl(bench_path)
    if args.bucket:
        buckets = {b.strip() for b in args.bucket.split(",") if b.strip()}
        items = [x for x in items if x.get("bucket") in buckets]
    if args.id_prefix:
        items = [x for x in items if str(x.get("id", "")).startswith(args.id_prefix)]
    if not items:
        raise SystemExit(f"No items after filters in {bench_path}")

    required = {
        "mamay4": ["mamay4"],
        "qwen": ["qwen"],
        "qwen7": ["qwen7"],
        "mamay12": ["mamay12"],
        "lapa": ["lapa"],
        "aya": ["aya"],
        "router_matrix": ["mamay4", "lapa", "aya"],  # v2 rules: no qwen7
        "router_small": ["mamay4", "qwen"],
        "router_cascade": ["mamay4", "mamay12"],
        "router_cascade_micro": ["mamay4", "lapa", "aya", "qwen7"],
        "router_best": ["mamay4", "lapa", "aya"],  # no qwen7: code→mamay4, chat→aya
        "router_ensemble": ["mamay4", "lapa", "aya"],
        "all": ["mamay4", "lapa", "aya"],
    }[args.mode]
    if not args.skip_health:
        health_check(required)

    def pick_matrix(item):
        """Bake-off matrix router: different models per bucket."""
        d = route_intent(item["prompt"])
        alias = d.model if d.model in ROUTER_MODELS else "mamay4"
        base, model = BACKENDS[alias]
        return alias, base, model, {
            "model": alias,
            "intent": d.intent,
            "reason": d.reason,
            "backend_model": model,
            "api_base": base,
        }

    def pick_best(item):
        """Oracle: gold suite bucket → empirically best specialist on v4."""
        bucket = str(item.get("bucket") or "chat")
        alias = BEST_BY_BUCKET.get(bucket, "mamay4")
        if alias not in BACKENDS:
            raise SystemExit(f"BEST_BY_BUCKET alias unknown: {alias}")
        base, model = BACKENDS[alias]
        return alias, base, model, {
            "model": alias,
            "intent": bucket,
            "reason": f"oracle best-by-bucket {bucket}→{alias}",
            "backend_model": model,
            "api_base": base,
        }

    def pick_legacy_small(item):
        """Old S1 ablation: code→qwen, else→mamay4 (known to lose on bake-off)."""
        d = route_intent(item["prompt"])
        alias = "qwen" if d.intent == "code" else "mamay4"
        base, model = BACKENDS[alias]
        return alias, base, model, {
            "model": alias,
            "intent": d.intent,
            "reason": f"legacy small: {d.intent}→{alias}",
            "backend_model": model,
            "api_base": base,
        }

    def pick_fixed(alias: str):
        def _pick(item):
            base, model = BACKENDS[alias]
            return alias, base, model, {
                "model": alias,
                "intent": "fixed",
                "reason": f"always {alias}",
                "backend_model": model,
                "api_base": base,
            }

        return _pick

    global SEED, MAX_TOKENS
    base_seed = SEED
    if args.max_tokens is not None:
        MAX_TOKENS = int(args.max_tokens)

    for rep in range(1, max(1, args.repeats) + 1):
        if args.seed_jitter:
            SEED = base_seed + rep - 1
        else:
            SEED = base_seed

        bench_tag = bench_path.stem
        if args.id_prefix:
            bench_tag += (
                "_zno" if args.id_prefix.startswith("zno") else f"_{args.id_prefix.rstrip('-')}"
            )
        elif args.bucket:
            bench_tag += "_" + "_".join(
                sorted({b.strip() for b in args.bucket.split(",") if b.strip()})
            )
        bench_tag += f"_t{TEMPERATURE:g}_s{SEED}"
        if args.repeats > 1:
            bench_tag += f"_rep{rep}"

        print(
            f"bench={bench_path} n={len(items)} mode={args.mode} rep={rep}/{args.repeats} "
            f"align={args.align_prompt} "
            f"decoding={{temperature={TEMPERATURE}, seed={SEED}, max_tokens={MAX_TOKENS}}}"
        )

        kw: dict = {
            "vram_interval_s": args.vram_interval,
            "align_prompt": args.align_prompt,
        }
        if args.out_dir:
            kw["out_dir"] = Path(args.out_dir)
            if not kw["out_dir"].is_absolute():
                kw["out_dir"] = ROOT / kw["out_dir"]

        if args.mode in ("all", "mamay4"):
            run_system("mamay4", pick_fixed("mamay4"), items, bench_tag, **kw)
        if args.mode in ("all", "router_matrix"):
            sys_name = (
                "router_matrix_v2_fewshot"
                if args.align_prompt == "fewshot"
                else (
                    "router_matrix_v2_clarified"
                    if args.align_prompt == "clarified"
                    else "router_matrix_v2"
                )
            )
            run_system(sys_name, pick_matrix, items, bench_tag, **kw)
        if args.mode == "router_best":
            sys_name = (
                "router_best_fewshot"
                if args.align_prompt == "fewshot"
                else (
                    "router_best_clarified"
                    if args.align_prompt == "clarified"
                    else "router_best"
                )
            )
            run_system(sys_name, pick_best, items, bench_tag, **kw)
        if args.mode == "router_small":
            run_system("router_small", pick_legacy_small, items, bench_tag, **kw)
        if args.mode in ("all", "mamay12"):
            run_system("mamay12", pick_fixed("mamay12"), items, bench_tag, **kw)
        if args.mode == "qwen":
            run_system("qwen", pick_fixed("qwen"), items, bench_tag, **kw)
        if args.mode == "qwen7":
            run_system("qwen7", pick_fixed("qwen7"), items, bench_tag, **kw)
        if args.mode in ("all", "lapa"):
            run_system("lapa", pick_fixed("lapa"), items, bench_tag, **kw)
        if args.mode in ("all", "aya"):
            run_system("aya", pick_fixed("aya"), items, bench_tag, **kw)
        if args.mode == "router_cascade":
            # Legacy: matrix first hop, escalate to Mamay-12B on empty/hedge/no-digit.
            run_system(
                "router_cascade",
                pick_matrix,
                items,
                bench_tag,
                cascade=True,
                cascade_kind="legacy",
                **kw,
            )
        if args.mode == "router_cascade_micro":
            micro_name = (
                "router_cascade_micro_fewshot"
                if args.align_prompt == "fewshot"
                else "router_cascade_micro"
            )
            run_system(
                micro_name,
                pick_matrix,
                items,
                bench_tag,
                cascade=True,
                cascade_kind="micro",
                **kw,
            )
        if args.mode == "router_ensemble":
            ens_name = (
                "router_ensemble_vote_fewshot"
                if args.align_prompt == "fewshot"
                else "router_ensemble_vote"
            )
            run_system(
                ens_name,
                pick_matrix,
                items,
                bench_tag,
                ensemble=True,
                **kw,
            )


if __name__ == "__main__":
    main()
