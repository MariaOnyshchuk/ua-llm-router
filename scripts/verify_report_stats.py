#!/usr/bin/env python3
"""Recompute every headline statistic quoted in the status report from raw artifacts.

Read-only: reads results/*.jsonl and benchmarks/*.jsonl, prints a verification table.
Exists so numbers in docs/ can be re-derived on demand rather than trusted.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BENCH = ROOT / "benchmarks"

SYSTEMS = ["mamay4", "router_small", "mamay12", "qwen"]
LABEL = {
    "mamay4": "S0 Mamay-4B",
    "router_small": "S1 Router",
    "mamay12": "S2 Mamay-12B",
    "qwen": "Qwen-3B",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_detail(name: str) -> dict[str, dict[str, dict]]:
    """system -> id -> scored row"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in load_jsonl(RES / name):
        out[str(r["system"])][str(r["id"])] = r
    return out


def bootstrap_ci(vals: list[float], n: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(vals)
    means = []
    for _ in range(n):
        means.append(sum(vals[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    detail = load_detail("scores_mixed_ua_v2_detail.jsonl")

    print("=" * 78)
    print("1. OVERALL QUALITY AND BOOTSTRAP CI (n=160, 20000 resamples)")
    print("=" * 78)
    ids = sorted(detail["mamay4"])
    print(f"{'system':14} {'n':>4} {'mean':>8} {'95% CI':>22}")
    scores: dict[str, list[float]] = {}
    for s in SYSTEMS:
        vals = [float(detail[s][i]["score"]) for i in ids]
        scores[s] = vals
        lo, hi = bootstrap_ci(vals)
        print(f"{LABEL[s]:14} {len(vals):4d} {statistics.mean(vals):8.4f}   [{lo:.4f}, {hi:.4f}]")

    print()
    print("=" * 78)
    print("2. PAIRED BOOTSTRAP DIFFERENCES")
    print("=" * 78)
    pairs = [
        ("router_small", "mamay4"),
        ("router_small", "mamay12"),
        ("mamay4", "mamay12"),
        ("qwen", "mamay4"),
    ]
    print(f"{'comparison':34} {'mean diff':>10} {'95% CI':>22}  verdict")
    for a, b in pairs:
        diffs = [scores[a][i] - scores[b][i] for i in range(len(ids))]
        lo, hi = bootstrap_ci(diffs)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        name = f"{LABEL[a]} - {LABEL[b]}"
        print(f"{name:34} {statistics.mean(diffs):10.4f}   [{lo:+.4f}, {hi:+.4f}]  {sig}")

    print()
    print("=" * 78)
    print("3. WIN / LOSS / TIE vs S2 Mamay-12B (item level)")
    print("=" * 78)
    for s in ["mamay4", "router_small", "qwen"]:
        wins = sum(1 for i in range(len(ids)) if scores[s][i] > scores["mamay12"][i])
        losses = sum(1 for i in range(len(ids)) if scores[s][i] < scores["mamay12"][i])
        ties = len(ids) - wins - losses
        print(f"{LABEL[s]:14} {wins:3d} wins / {losses:3d} losses / {ties:3d} ties")

    print()
    print("=" * 78)
    print("4. ROUTING DISTRIBUTION (where did S1 actually send prompts?)")
    print("=" * 78)
    routed: dict[str, int] = defaultdict(int)
    bucket_backend: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in ids:
        r = detail["router_small"][i].get("router") or {}
        m = str(r.get("model"))
        routed[m] += 1
        bucket_backend[str(detail["router_small"][i]["bucket"])][m] += 1
    for m, c in sorted(routed.items(), key=lambda kv: -kv[1]):
        print(f"  {m:12} {c:4d} prompts")
    print("  by bucket:")
    for b in sorted(bucket_backend):
        inner = ", ".join(f"{m}={c}" for m, c in sorted(bucket_backend[b].items()))
        print(f"    {b:11} {inner}")

    print()
    print("=" * 78)
    print("5. NONDETERMINISM: S0 vs S1 on prompts routed to the SAME backend")
    print("=" * 78)
    same = [
        i for i in ids
        if (detail["router_small"][i].get("router") or {}).get("model") == "mamay4"
    ]
    disagree = [i for i in same if scores["router_small"][ids.index(i)] != scores["mamay4"][ids.index(i)]]
    md = statistics.mean(
        [scores["router_small"][ids.index(i)] - scores["mamay4"][ids.index(i)] for i in same]
    )
    print(f"  identical-backend prompts : {len(same)} of {len(ids)}")
    print(f"  scores differ on          : {len(disagree)} items")
    print(f"  mean difference on those  : {md:+.4f}")
    print(f"  example flipped ids       : {', '.join(sorted(disagree)[:8])}")

    print()
    print("=" * 78)
    print("6. TEXT-LEVEL DIVERGENCE + ROUTER OVERHEAD (external_ua_v2, 96 matched prompts)")
    print("=" * 78)
    direct = {r["id"]: r for r in load_jsonl(RES / "mamay4_external_ua_v2_20260727T082941Z.jsonl")}
    viar = {r["id"]: r for r in load_jsonl(RES / "router_small_external_ua_v2_20260727T083059Z.jsonl")}
    common = sorted(set(direct) & set(viar))
    difftext = [i for i in common if (direct[i].get("content") or "") != (viar[i].get("content") or "")]
    dl = [float(viar[i]["latency_ms"]) - float(direct[i]["latency_ms"]) for i in common]
    print(f"  matched prompts        : {len(common)}")
    print(f"  different output text  : {len(difftext)}")
    print(f"  mean latency direct    : {statistics.mean([float(direct[i]['latency_ms']) for i in common]):8.1f} ms")
    print(f"  mean latency via router: {statistics.mean([float(viar[i]['latency_ms']) for i in common]):8.1f} ms")
    print(f"  mean per-item delta    : {statistics.mean(dl):+8.1f} ms")
    print(f"  median per-item delta  : {statistics.median(dl):+8.1f} ms")

    print()
    print("=" * 78)
    print("7. PUBLIC DATASET SUBSETS")
    print("=" * 78)
    zno = [i for i in ids if i.startswith("zno-")]
    ual = [i for i in ids if i.startswith("ualign-")]
    flo = [i for i in ids if i.startswith("flores-")]
    en_uk = [i for i in flo if "en-uk" in i or i.endswith("-enuk")]
    uk_en = [i for i in flo if i not in en_uk]
    print(f"  subset sizes: ZNO={len(zno)}  UAlign={len(ual)}  FLORES={len(flo)}"
          f" (en->uk={len(en_uk)}, uk->en={len(uk_en)})")
    print(f"\n  {'system':14} {'ZNO acc':>16} {'UAlign acc':>16} {'FLORES':>9} {'en>uk':>8} {'uk>en':>8}")
    for s in SYSTEMS:
        zc = sum(float(detail[s][i]["score"]) for i in zno)
        uc = sum(float(detail[s][i]["score"]) for i in ual)
        fm = statistics.mean([float(detail[s][i]["score"]) for i in flo]) if flo else 0
        e = statistics.mean([float(detail[s][i]["score"]) for i in en_uk]) if en_uk else 0
        u = statistics.mean([float(detail[s][i]["score"]) for i in uk_en]) if uk_en else 0
        print(f"  {LABEL[s]:14} {zc:5.0f}/{len(zno)} = {zc/max(1,len(zno)):.3f}"
              f" {uc:5.0f}/{len(ual)} = {uc/max(1,len(ual)):.3f}"
              f" {fm:9.4f} {e:8.4f} {u:8.4f}")

    print()
    print("=" * 78)
    print("8. TOKENS PER PROMPT")
    print("=" * 78)
    print("  NOTE: verbosity must be read off completion_tokens. total_tokens mixes in")
    print("  prompt_tokens, which differ per model because the chat templates and")
    print("  tokenizers differ -- Qwen spends ~142 prompt tokens where Mamay spends ~89.")
    runs = {
        "mamay4": ["mamay4_mixed_ua_v1_20260727T070125Z.jsonl", "mamay4_external_ua_v2_20260727T082941Z.jsonl"],
        "router_small": ["router_small_mixed_ua_v1_20260727T070416Z.jsonl", "router_small_external_ua_v2_20260727T083059Z.jsonl"],
        "mamay12": ["mamay12_mixed_ua_v1_20260727T070730Z.jsonl", "mamay12_external_ua_v2_20260727T083217Z.jsonl"],
        "qwen": ["qwen_mixed_ua_v1_20260727T070936Z.jsonl", "qwen_external_ua_v2_20260727T083406Z.jsonl"],
    }
    print(f"\n  {'system':14} {'rows':>5} {'prompt':>9} {'completion':>11} {'total':>9}")
    for s, files in runs.items():
        prompt_t, comp_t, total_t, n = [], [], [], 0
        for f in files:
            for r in load_jsonl(RES / f):
                n += 1
                u = r.get("usage") or {}
                for key, acc in (
                    ("prompt_tokens", prompt_t),
                    ("completion_tokens", comp_t),
                    ("total_tokens", total_t),
                ):
                    if u.get(key) is not None:
                        acc.append(float(u[key]))
        if not comp_t:
            print(f"  {LABEL[s]:14} {n:5d}  no usage field recorded")
            continue
        print(
            f"  {LABEL[s]:14} {n:5d} {statistics.mean(prompt_t):9.1f}"
            f" {statistics.mean(comp_t):11.1f} {statistics.mean(total_t):9.1f}"
        )

    print()
    print("=" * 78)
    print("9. HEALTH: http status / empty responses across the 8 runs of 2026-07-27")
    print("=" * 78)
    for s, files in runs.items():
        bad = empty = tot = 0
        for f in files:
            for r in load_jsonl(RES / f):
                tot += 1
                if r.get("http_status") != 200:
                    bad += 1
                if not (r.get("content") or "").strip():
                    empty += 1
        print(f"  {LABEL[s]:14} n={tot:4d}  non-200={bad}  empty={empty}")


if __name__ == "__main__":
    main()
