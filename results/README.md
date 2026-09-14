# Results index

Local laptop copy. **Full `mixed_ua_v3` JSONL runs (18 × 182 rows) live on the lab**  
`ucu-lab-2240:~/Diploma/results/` — only summaries / matrices were pulled here for weeks 3–4.

| Folder | When | What |
|--------|------|------|
| [`week_1_wiring_smoke/`](week_1_wiring_smoke/) | ~25 Jul | First serving + router wiring. Small smoke runs (dual, S0/S1/S2, early Lapa). Not thesis tables. |
| [`week_2_v1_and_external/`](week_2_v1_and_external/) | ~26–27 Jul | `mixed_ua_v1` (64) + `external_ua_v2` (FLORES/UAlign). First real multi-system scores. |
| [`week_3_v2_and_pinned_v3/`](week_3_v2_and_pinned_v3/) | ~28–29 Jul | `mixed_ua_v2` scores; T=0 Mamay-4B isolation; **pinned v3 mean±sd** for Mamay4 / Mamay12 / rules router / cascade. |
| [`week_4_specialist_bakeoff/`](week_4_specialist_bakeoff/) | ~3 Aug | **Authoritative bake-off matrix**: + Lapa alone + Qwen alone. Use this for router decisions. |
| [`week_5_aya_qwen7/`](week_5_aya_qwen7/) | ~7 Aug | Aya alone on `mixed_ua_v4` (mean±sd) + Qwen-Coder-7B **code-only** bake-off. |
| [`week_5_router_v4/`](week_5_router_v4/) | ~7 Aug | **Matrix router** on v4 (192×3): scores, JSONL, routing/success/cost stats; Aya rescored side-by-side. |
| [`week_5_alignment_bakeoff/`](week_5_alignment_bakeoff/) | ~8 Aug | Alignment slice: Lapa vs Mamay-4B × baseline / clarified / few-shot (`scores_alignment_bakeoff.json`). |
| [`week_6_v4_solos/`](week_6_v4_solos/) | 20 Aug | **Mamay-4B + Lapa** full `mixed_ua_v4` 192×3 baseline; `scores_mean_sd_v4_solos.json`. |
| [`week_6_router_fewshot/`](week_6_router_fewshot/) | 20 Aug | Matrix router ×3 with social few-shot + leakage-fixed rules (0.842). |
| [`week_6_cascade_micro/`](week_6_cascade_micro/) | 20 Aug | Social-2 → Mamay-4B retry; escalate 1.56%, no quality gain. |
| [`week_7_router_best/`](week_7_router_best/) | 21 Aug | Gold-bucket best specialists on v4: 0.848 / p50 633 vs rules 0.842 / 441. |
| [`week_7_rules_v2/`](week_7_rules_v2/) | 21 Aug | Rules v2 (chat→Aya, code→Mamay-4B) on v4: **0.848**, matches oracle. |
| [`week_7_v5/`](week_7_v5/) | 21 Aug | v4+32 HumanEval (224); rules v2 overall 0.796 (HE 19/32; chat hurt by max_tokens=512). |
| [`week_7_ensemble/`](week_7_ensemble/) | 31 Aug | Majority vote on alignment+ZNO; overall **0.843** vs rules v2 0.848; 3 flips net −1. |
| [`week_8_quant/`](week_8_quant/) | 31 Aug | Rules v2 on bitsandbytes 4-bit Mamay/Lapa/Aya: **0.830**, slower than bf16 0.848. |
| [`week_10_composite/`](week_10_composite/) | 1 Sep | Authoritative composite 36×3 summary: oracle **0.882**, one-hop **0.812**, hybrid planner **0.768**. |
| [`_empty_failed_stubs/`](_empty_failed_stubs/) | — | Zero-byte / aborted runs. Safe to ignore or delete. |

Wide CSVs (open in Excel): [`tables/v4_model_bucket_quality.csv`](tables/v4_model_bucket_quality.csv), [`tables/v4_alignment_prompt_grid.csv`](tables/v4_alignment_prompt_grid.csv), [`tables/composite_v1.csv`](tables/composite_v1.csv), [`tables/v3_specialist_bakeoff.csv`](tables/v3_specialist_bakeoff.csv). Screening means: [`tables/v6_screen_not_for_claims.csv`](tables/v6_screen_not_for_claims.csv) — do not cite. Frozen-v6 **preview** (screening scores on the 2414 curated ids, not the 3× bake-off): [`tables/v6_preview_from_screen.csv`](tables/v6_preview_from_screen.csv), [`tables/v6_preview_model_bucket.csv`](tables/v6_preview_model_bucket.csv), [`tables/v6_preview_items.csv`](tables/v6_preview_items.csv). Rebuild with `python scripts/export_results_tables.py`. Long-format ledger: [`progress_ledger.csv`](progress_ledger.csv).

## Start here for claims

1. **Bucket winners / route advice (v3):** `week_4_specialist_bakeoff/specialist_matrix_bakeoff.json`
2. **Mean ± sd all systems on v3 (incl. Lapa/Qwen):** `week_4_specialist_bakeoff/scores_mean_sd_bakeoff.json`
3. **Same-suite v4 solos + few-shot router:** `week_6_v4_solos/`, `week_6_router_fewshot/`, `week_6_cascade_micro/`
4. **Router vs Aya on v4 (baseline prompt):** `week_5_router_v4/scores_mean_sd_router_v4.json` (+ `scores_mean_sd_aya_v4_rescored.json`)
5. **Alignment prompt/model ablation:** `week_5_alignment_bakeoff/scores_alignment_variants.json`
6. **Narrative summary:** `docs/key_takeaways.md`
6b. **Week-6 tables:** `docs/week_6_takeaways.md`
6c. **Thesis conclusion:** `docs/conclusion.md`
6d. **Composite router protocol/results:** `docs/week_10_composite.md`
7. **Earlier v3 without Lapa/Qwen alone:** `week_3_v2_and_pinned_v3/scores_mean_sd_v3.json`

## Lab paths for raw v3 JSONL

On `ucu-lab-2240`:

```text
~/Diploma/results/{mamay4,mamay12,qwen,lapa,router_small,router_cascade}_mixed_ua_v3_t0_s42_rep{1,2,3}_*.jsonl
~/Diploma/results/scored_reps/
~/Diploma/results/punchlist_*.out   # week 3 pipeline log
~/Diploma/results/bakeoff_ql_*.out  # week 4 pipeline log
```
