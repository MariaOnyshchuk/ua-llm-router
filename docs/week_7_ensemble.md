# Week 7 — S3 discrete ensemble (majority vote)

*31 Aug 2026. Suite: `mixed_ua_v4_balanced.jsonl` (192), T=0, seed=42, **1 repeat**, max_tokens=256, few-shot alignment.*

Rules v2 first hop, then **majority vote only** on votable discrete labels: all **alignment** (32) + **ZNO** (`id` starts with `zno-`, 24). Voters: Lapa, Mamay-4B, Aya. Tie-break = first hop. Open knowledge / chat / code / translate stay single-call.

## Result vs rules v2 (0.848)

| System | Overall | p50 | knowledge | alignment | extra GPU-s | vote / flip |
|--------|--------:|----:|----------:|----------:|------------:|-------------|
| Rules v2 + few-shot (3×) | **0.848** | **633** | **0.750** | 0.750 | — | — |
| **Ensemble vote + few-shot (1×)** | **0.843** | 1201 | **0.688** | **0.781** | **+59** (354 vs ~295) | 56/192 voted, **3 flipped** |

Chat / code / instruct match v2 (1.000 / 0.984 / 0.781). Translate 0.822 vs 0.820 is run noise, not voting.

The three flips **are** the quality change:

| id | First hop (Lapa) | Majority | Outcome |
|----|------------------|----------|---------|
| `zno-geo-01-1-3927` | Б (correct) | Mamay+Aya → В | **hurt** |
| `zno-geo-03-3-4002` | А (correct) | Mamay+Aya → Б | **hurt** |
| `ualign-social-0451` | 1 (wrong) | Mamay+Aya → 2 | **helped** |

Net **−1** discrete item. Specialists disagree on ZNO; majority followed the two weaker knowledge models and overwrote Lapa.

Vote rate 29%; flip rate 5% of votes (3/56). Wall p50 doubles because ZNO/alignment now pay **three** sequential HTTP calls (latency summed).

**Verdict:** same as the micro-cascade — **not worth shipping**. Rules v2 stays the product. Ensemble is a measured upper bound on **votable** items only, and that bound is *below* the specialist first hop.

Mode: `python scripts/run_small_router_cluster.py --mode router_ensemble --align-prompt fewshot`. Helpers: `router/ensemble.py`.

Artifacts: `results/week_7_ensemble/`.
