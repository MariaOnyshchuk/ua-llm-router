# Week 7 — oracle “best router” on v4

*21 Aug 2026. Suite: `mixed_ua_v4_balanced.jsonl` (192×3), T=0, seed=42, few-shot alignment.*

This run uses the suite’s gold `bucket` label, not regex: **chat→Aya**, **code→Mamay-4B**, knowledge/alignment→Lapa, instruct→Mamay-4B, translate→Aya. Qwen unused (3 GPUs).

| System | Overall | p50 | chat | code |
|--------|--------:|----:|-----:|-----:|
| Oracle best + few-shot | **0.848** | 633 ms | **1.000** | **0.984** |
| Rules v1 + few-shot | 0.842 | **441** | 0.984 | 0.969 |
| Rules v2 + few-shot (same map in regex) | **0.848** | 633 | **1.000** | **0.984** |

Headroom vs rules v1 is those two swaps. Rules v2 later **matches** this oracle on v4.

Artifacts: `results/week_7_router_best/`.
