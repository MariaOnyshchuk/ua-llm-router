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
from router.client import sync_chat  # noqa: E402
from router.intent_rules import BEST_BY_BUCKET, ROUTER_MODELS, route_intent  # noqa: E402
from scripts.alignment_prompt_variants import apply_alignment_variant  # noqa: E402
from router.format_guard import apply as apply_format_guard  # noqa: E402
from scripts.bucket_prompt_variants import apply_bucket_fewshot, parse_buckets  # noqa: E402
from router.wiki_search import WikiClient, WikiConfig, wiki_augment  # noqa: E402

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
    align_prompt: str = "baseline",
    bucket_fewshot: frozenset[str] = frozenset(),
    format_guard: bool = False,
    wiki: WikiClient | None = None,
    out_dir: Path | None = None,
) -> Path:
    if wiki is not None:
        # Wikipedia passages change the prompt, so it is a different system
        name = f"{name}_wiki"
    if format_guard:
        # the guard changes answers, so it is a different system in file names and scores
        name = f"{name}_fg"
    if bucket_fewshot:
        # keep runs with different prompts apart in file names and in the system column
        name = f"{name}_bfs-{'-'.join(sorted(bucket_fewshot))}"
    dest = Path(out_dir) if out_dir else (ROOT / "results")
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = dest / f"{name}_{bench_tag}_{stamp}.jsonl"
    partial_out = out.with_suffix(out.suffix + ".partial")

    sampler = VramSampler(interval_s=vram_interval_s)
    sampler.start()
    vram_before = sampler.samples[0]
    latencies: list[float] = []
    wiki_used = 0
    wiki_errors = 0

    with httpx.Client(timeout=180.0) as client, partial_out.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(items, 1):
            tagged = apply_alignment_variant(item, align_prompt)
            # routing sees the item without examples; only the text sent to the model changes
            prompt = apply_bucket_fewshot(tagged, bucket_fewshot)["prompt"]
            alias, base, model, router_meta = pick_backend(tagged)
            wiki_meta = None
            if wiki is not None:
                # the same rules-v2 gate for every system, so solo models and the router see
                # the same passages; queries come from the item text without few-shot wrapping
                gate_intent = route_intent(str(item.get("prompt") or "")).intent
                prompt, wiki_meta = wiki_augment(
                    prompt, gate_intent, wiki, query_prompt=str(item.get("prompt") or "")
                )
                wiki_meta.pop("config", None)
                for passage in wiki_meta.get("passages") or []:
                    passage.pop("text", None)  # the text is already in the prompt
                wiki_used += bool(wiki_meta.get("used"))
                wiki_errors += bool(wiki_meta.get("errors"))
            result = chat(client, base, model, prompt)
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
                "ifeval_instruction_id_list": tagged.get("ifeval_instruction_id_list"),
                "ifeval_kwargs": tagged.get("ifeval_kwargs"),
                "system": name,
                "prompt_variant": align_prompt,
                "bucket_fewshot": sorted(bucket_fewshot),
                "model_requested": router_meta.get("model", alias),
                "router": router_meta,
                "wiki": wiki_meta,
                **result,
            }
            if format_guard:
                # constraints are read from the item's own request, not from the few-shot wrapper
                guarded, repairs = apply_format_guard(str(item.get("prompt") or ""), row.get("content") or "")
                row["content_raw"] = row.get("content") or ""
                row["content"] = guarded
                row["format_guard"] = repairs
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            preview = (result.get("content") or "")[:70].replace("\n", " ")
            print(
                f"[{i}/{len(items)}] {item['id']} → {row['model_requested']} "
                f"{result['http_status']} {result['latency_ms']}ms | {preview}"
            )
            if not result.get("ok"):
                sampler.stop()
                raise RuntimeError(
                    f"eval backend failed at item {item['id']} via {alias}: "
                    f"HTTP {result.get('http_status')} {result.get('error')}. "
                    f"Partial output retained at {partial_out}"
                )

    vram_stats = sampler.stop()
    partial_out.replace(out)
    gpu_seconds_total = sum(latencies) / 1000.0
    meta = {
        "system": name,
        "bench_tag": bench_tag,
        "n": len(items),
        "wrote": str(out),
        "prompt_variant": align_prompt,
        "bucket_fewshot": sorted(bucket_fewshot),
        "decoding": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
        },
        "wiki_rag": None
        if wiki is None
        else {
            "used": wiki_used,
            "use_rate": round(wiki_used / max(1, len(items)), 4),
            "items_with_errors": wiki_errors,
            "offline": wiki.cfg.offline,
            "cache_dir": str(wiki.cfg.cache_dir),
            "api_calls": wiki.api_calls,
            "cache_hits": wiki.cache_hits,
            "config": {k: v for k, v in vars(wiki.cfg).items() if k != "cache_dir"},
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
        "mamay27",
        "qwen",
        "qwen7",
        "lapa",
        "aya",
        "router_matrix",
        "router_small",
        "router_best",  # gold-bucket → v4 specialist winners (oracle, not rules)
        "router_knn",
        "router_clf",
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
        "--bucket-fewshot",
        default="",
        help="Comma list of buckets (chat,code,translate,instruct,knowledge) or 'all': "
        "put two worked examples before the prompt. Alignment uses --align-prompt.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Directory for JSONL + meta (default: results/).",
    )
    p.add_argument(
        "--bench",
        default="benchmarks/tasks_mixed_192.jsonl",
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
        "--format-guard",
        action="store_true",
        help="Repair explicit surface constraints of the request in each answer (router/format_guard.py). "
        "The system name gets the suffix _fg; the model's own text is kept in content_raw.",
    )
    p.add_argument(
        "--wiki-rag",
        action="store_true",
        help="Put Ukrainian Wikipedia passages in front of knowledge questions (router/wiki_search.py). "
        "The system name gets the suffix _wiki.",
    )
    p.add_argument(
        "--wiki-cache",
        default=".cache/wiki",
        help="Directory with cached Wikipedia API answers (default .cache/wiki).",
    )
    p.add_argument(
        "--wiki-offline",
        action="store_true",
        help="Use only the cache; build it first with scripts/build_wiki_cache.py. Needed on nodes "
        "without internet and for repeatable runs.",
    )
    p.add_argument(
        "--split",
        default="",
        help="Keep only rows with this value in their 'split' field (mixed_ua_v7: dev or test).",
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
    if args.split:
        items = [x for x in items if x.get("split") == args.split]
    if not items:
        raise SystemExit(f"No items after filters in {bench_path}")

    required = {
        "mamay4": ["mamay4"],
        "qwen": ["qwen"],
        "qwen7": ["qwen7"],
        "mamay12": ["mamay12"],
        "mamay27": ["mamay27"],
        "lapa": ["lapa"],
        "aya": ["aya"],
        "router_matrix": ["mamay4", "lapa", "aya"],  # v2 rules: no qwen7
        "router_small": ["mamay4", "qwen"],
        "router_best": ["mamay4", "lapa", "aya"],  # no qwen7: code→mamay4, chat→aya
        "router_knn": ["mamay4", "lapa", "aya"],
        "router_clf": ["mamay4", "lapa", "aya"],
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

    def pick_learned(profile: str):
        def _pick(item):
            d = route_intent(item["prompt"], profile=profile)
            alias = d.model if d.model in ROUTER_MODELS else "mamay4"
            if alias not in BACKENDS:
                alias = "mamay4"
            base, model = BACKENDS[alias]
            return alias, base, model, {
                "model": alias,
                "intent": d.intent,
                "reason": d.reason,
                "backend_model": model,
                "api_base": base,
                "profile": profile,
            }

        return _pick

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
        if args.split:
            bench_tag += f"_{args.split}"
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
            f"align={args.align_prompt} bucket_fewshot={args.bucket_fewshot or 'none'} "
            f"decoding={{temperature={TEMPERATURE}, seed={SEED}, max_tokens={MAX_TOKENS}}}"
        )

        kw: dict = {
            "vram_interval_s": args.vram_interval,
            "align_prompt": args.align_prompt,
            "bucket_fewshot": parse_buckets(args.bucket_fewshot),
            "format_guard": args.format_guard,
        }
        if args.wiki_rag:
            cache = Path(args.wiki_cache)
            kw["wiki"] = WikiClient(
                WikiConfig(cache_dir=cache if cache.is_absolute() else ROOT / cache, offline=args.wiki_offline)
            )
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
        if args.mode == "mamay27":
            run_system("mamay27", pick_fixed("mamay27"), items, bench_tag, **kw)
        if args.mode == "qwen":
            run_system("qwen", pick_fixed("qwen"), items, bench_tag, **kw)
        if args.mode == "qwen7":
            run_system("qwen7", pick_fixed("qwen7"), items, bench_tag, **kw)
        if args.mode in ("all", "lapa"):
            run_system("lapa", pick_fixed("lapa"), items, bench_tag, **kw)
        if args.mode in ("all", "aya"):
            run_system("aya", pick_fixed("aya"), items, bench_tag, **kw)
        if args.mode == "router_knn":
            run_system("router_knn", pick_learned("knn"), items, bench_tag, **kw)
        if args.mode == "router_clf":
            run_system("router_clf", pick_learned("clf"), items, bench_tag, **kw)


if __name__ == "__main__":
    main()
