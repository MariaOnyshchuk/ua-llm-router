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

**Headline pinned numbers (do not mix suites):**

- Rules v2 + few-shot on `mixed_ua_v4` (192×3): **0.848**
- Same router on bitsandbytes 4-bit (week 8): **0.830**
- Composite 36×3 (week 10): oracle **0.882**, one-hop **0.812**, hybrid planner **0.768**
- `mixed_ua_v5` (+HumanEval) **0.796** is a harder suite, not a regression vs 0.848
- Frozen `mixed_ua_v6` (2414 items, 1 pass, T=0, seed=42): dedicated solos in `v6_solos/` — Mamay-12B **0.695**, Lapa **0.669**, Aya **0.667**, Mamay-4B **0.665**, Qwen-7B **0.594**. Do not cite the incomplete Mamay-12B screening JSONL. Do not mix screening-sliced scores with this table.
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
