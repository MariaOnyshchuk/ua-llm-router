---
license: cc-by-4.0
pretty_name: UA Specialist Router — evaluation progress
language:
  - uk
  - en
task_categories:
  - text-generation
tags:
  - ukrainian
  - llm-routing
  - evaluation
  - diploma
size_categories:
  - n<1K
---

# UA Specialist Router — evaluation progress

Score tables, routing stats, and run metadata from the diploma project
[MariaOnyshchuk/ua-llm-router](https://github.com/MariaOnyshchuk/ua-llm-router):
a rules-based router over open Ukrainian specialists (Mamay-4B, Lapa-12B, Aya Expanse 8B, Qwen-Coder).

This dataset is the **progress log of pinned JSON summaries**, not a dump of every generation.

## What is included

| Path | Contents |
|------|----------|
| `progress_ledger.csv` | Flattened metric rows across weeks (best table for browsing) |
| `week_*/scores*.json`, `scores_mean_sd_*.json` | Suite-level quality and latency |
| `week_*/specialist_matrix*.json` | Per-bucket bake-off used to set routes |
| `week_*/*.meta.json` | Decoding (`T=0`, `seed=42`), model ids, GPU-seconds |
| `week_*/routing_stats*.json` | Traffic mix for router runs |
| `week_*/scores_detail.jsonl` | Item-level scores (ids + scores, no prompts) |
| `v6_solos/scores.json` | Frozen `mixed_ua_v6` 1-pass dedicated solos (five models) |
| `v6_solos/*.meta.json` | Decoding, GPU-seconds, model ids for those runs |
| `tables/v6_solos_1pass.csv` | Same numbers as a wide CSV |
| `tables/v6_preview_from_screen.csv` | Screening-sliced preview — do not mix with `v6_solos` |
| `tables/a2_planner_grid*.csv` | Planning-only prompt × model screen on composite v1 |
| `tables/a2_agentcoma*.csv` | Planning-only AgentCoMa diagnostics (IDs and metrics; no prompts or generations) |
| `week_9_baselines/` | Mamay-12B alone on `mixed_ua_v4_balanced` (S2 baseline), 3 repeats: scores, item-level scores, run metadata |
| `week_11_pack_fp8/` | Rules v2 router with three specialists in online FP8 on one GPU, 3 repeats: scores, item-level scores, run metadata, `nvidia-smi` after load |
| `week_12_composite_v2_*/` | Composite v2 planning results (offline template selector on dev and test). `fixture/scores.json` is a code check, see the note below |
| `analysis/planner_split_v6_hybrid.json` | How often the Mamay-4B `hybrid` planner splits single-skill `mixed_ua_v6` requests: 400 sampled items, 100 per claim bucket, counts and Wilson 95% intervals, no prompts. A planner diagnostic, not a system comparison |
| `analysis/bootstrap_comparisons.json` | Paired item-level bootstrap (95% intervals) for router vs Mamay-12B comparisons on `mixed_ua_v4_balanced` |

**Headline pinned numbers (do not mix suites):**

- **Scorer note (29 Sep 2026):** from 4 Sep to 29 Sep `score_results.py` scored numeric-fact knowledge references (know-002 = 1991, know-004 = 24) as multiple-choice indexes, so every model got 0 on those two items (-0.0104 overall on v4). Fixed on 29 Sep. Rescored with the fixed scorer and published here: week 9 (Mamay-12B), week 11 (FP8 pack) and `v6_solos`. The 0.848 router and the week 6 solo numbers were scored before the bug and are unaffected. The week 6 and week 7 rescoring files made inside the bug window are not published; nothing from them is cited.
- Rules v2 + few-shot on `mixed_ua_v4` (192×3): **0.848**
- Mamay-12B alone on `mixed_ua_v4` (192×3, S2 baseline): **0.884** (rescored; 0.874 before the fix), p50 1352 ms, about 2.5 GPU-seconds per prompt. The router: 0.848, p50 633 ms, about 1.5 GPU-seconds per prompt, three resident GPUs. The 12B model is the better single-number quality result; the router is faster and cheaper per prompt.
- Same router with the three specialists in online FP8 on **one** GPU (week 11): **0.852** (rescored; 0.842 before the fix), p50 723 ms, 45 668 MiB resident. Knowledge is 0.750, the same as the bf16 router; the earlier drop to 0.688 came from the scorer bug.
- Paired item-level bootstrap, router (FP8 run) minus Mamay-12B: overall **-0.032**, 95% interval [-0.072, +0.008], not significant (unchanged by the rescoring: both systems gain the same two items). Instruct: -0.156, [-0.281, -0.031], significant. The FP8 run stands in for the 0.848 router because item-level scores for the bf16 router are not stored. `analysis/bootstrap_comparisons.json`
- Same router on bitsandbytes 4-bit (week 8): **0.830**
- Composite 36×3 (week 10): oracle **0.882**, one-hop **0.812**, hybrid planner **0.768**
- `mixed_ua_v5` (+HumanEval) **0.796** is a harder suite, not a regression vs 0.848
- Frozen `mixed_ua_v6` (2414 items, 1 pass, T=0, seed=42): dedicated solos in `v6_solos/` — Mamay-12B **0.695**, Lapa **0.670**, Aya **0.668**, Mamay-4B **0.666**, Qwen-7B **0.595**. Do not cite the incomplete Mamay-12B screening JSONL. Do not mix screening-sliced scores with this table.
- Composite v2 (200 items, train/dev/test): the constrained template selector matches the stored workflow on all 60 dev and all 60 test items (exact workflow match 1.000). This is a planning result only. **Specialist execution on the test split has not been run yet.**
- `week_12_composite_v2_test_bakeoff/fixture/scores.json` shows final scores of 1.0 for the oracle and template systems. Those runs use fixture specialists that return the expected answer, so they check the scoring and orchestration code. They say nothing about model quality and must not be cited as results.
- A2 planner tables are diagnostics, not end-task quality results. The AgentCoMa
  intent sequence is a project-defined diagnostic oracle, not benchmark gold.

## What is not included (on purpose)

- Raw generation JSONL (`prompt` / `content`) — those rows reprint ZNO-Eval, FLORES-200, UAlign, and HumanEval text.
- UA-Code-Bench / Eolymp problem statements (not redistributable).
- AgentCoMa prompts or translated derivatives (the source dataset is gated and
  does not publish a reusable dataset license).
- Empty failed stubs and composite smoke folders.

Prompts live in the original sources and in the GitHub repo’s `benchmarks/` samples. Cite those datasets if you rebuild the suites.

## Decoding

Unless a `*.meta.json` says otherwise: temperature `0.0`, seed `42`. Week 1 wiring smoke is unpinned and marked superseded in the ledger.

## Load

```python
from datasets import load_dataset

ledger = load_dataset("USERNAME/ua-llm-router-eval", data_files="progress_ledger.csv")
print(ledger["train"][0])
```

Replace `USERNAME` with the Hugging Face namespace that hosts this repo.

## Citation

```
@misc{onyshchuk-ua-llm-router-eval,
  title  = {UA Specialist Router evaluation progress},
  author = {Onyshchuk, Maria},
  year   = {2026},
  url    = {https://huggingface.co/datasets/USERNAME/ua-llm-router-eval}
}
```

Code and thesis notes: https://github.com/MariaOnyshchuk/ua-llm-router
