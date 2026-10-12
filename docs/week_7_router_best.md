# Week 7 — oracle “best router” on v4

> **Scorer note (13 Oct 2026).** On 29 Sep a scorer bug was fixed (numeric
> answers such as 1991 were read as multiple-choice indexes). Rescored runs:
> week 9 (Mamay-12B on v4: 0.874 → **0.884**) and week 11 (FP8 pack: 0.842 →
> **0.852**); in both, the whole change is in knowledge (0.688 → 0.750). The
> week 5–8 runs in this file (rules v1 and v2, the three specialists,
> micro-cascade, ensemble vote, 4-bit) were **not rescored**: their raw files
> are only on the lab. Those numbers are from before the fix and are probably
> about 0.01 low. All of these are v4 development numbers; claims come from
> `mixed_ua_v7` test.

*21 Aug 2026. Suite: `mixed_ua_v4_balanced.jsonl` (192×3), T=0, seed=42, few-shot alignment.*

This run uses the suite’s gold `bucket` label, not regex: **chat→Aya**, **code→Mamay-4B**, knowledge/alignment→Lapa, instruct→Mamay-4B, translate→Aya. Qwen unused (3 GPUs).

| System | Overall | p50 | chat | code |
|--------|--------:|----:|-----:|-----:|
| Oracle best + few-shot | **0.848** | 633 ms | **1.000** | **0.984** |
| Rules v1 + few-shot | 0.842 | **441** | 0.984 | 0.969 |
| Rules v2 + few-shot (same map in regex) | **0.848** | 633 | **1.000** | **0.984** |

Headroom vs rules v1 is those two swaps. Rules v2 later **matches** this oracle on v4.

Artifacts: `results/week_7_router_best/`.
