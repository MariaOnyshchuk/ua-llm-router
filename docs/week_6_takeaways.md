# Week 6 takeaways — same-suite v4 solos, few-shot router, micro-cascade

*Lab: `ucu-lab-2240`, 20 Aug 2026. Suite: `benchmarks/mixed_ua_v4_balanced.jsonl` (192 = 32×6). Decoding: T=0, seed=42, 3 repeats. HTTP 192/192 on all week-6 JSONL.*

This week filled the missing **same-suite** table: Mamay-4B and Lapa solos on v4, then the matrix router with **social few-shot** and **leakage-fixed rules**, then a **micro-cascade** (no Mamay-12B).

---

## What we ran

| Phase | System | Prompt | GPUs | Wall (approx) |
|-------|--------|--------|-----:|---------------|
| A | Mamay-4B alone, Lapa alone | baseline | 2 then same | ~65 min total |
| B | `router_matrix_fewshot` | few-shot on social alignment only | 4 | ~15 min |
| C | `router_cascade_micro_fewshot` | same few-shot; social `2`/no-digit → one Mamay-4B retry | 4 | ~15 min |

Skipped: ensemble, Mamay-12B on v4, full 14k corpus.

---

## 1. Headline quality (mean over 3 repeats)

| System | Overall | sd | Macro | chat | code | translate | instruct | knowledge | alignment |
|--------|--------:|---:|------:|-----:|-----:|----------:|---------:|----------:|----------:|
| **Router + few-shot** (week 6) | **0.8425** | 0.0003 | 0.8425 | 0.984 | **0.969** | **0.821** | 0.781 | **0.750** | **0.750** |
| Router + micro-cascade (week 6) | 0.8423 | 0.0000 | 0.8423 | 0.984 | 0.969 | 0.820 | 0.781 | 0.750 | 0.750 |
| Router matrix baseline (7 Aug) | 0.8163 | 0.0000 | 0.8163 | 0.984 | **0.969** | 0.820 | 0.781 | **0.750** | 0.594 |
| Aya Expanse 8B alone | 0.7852 | 0.0003 | 0.7852 | **1.000** | 0.953 | **0.821** | 0.781 | 0.562 | 0.594 |
| Mamay-4B alone | 0.7617 | 0.0001 | 0.7617 | 0.984 | **0.984** | 0.758 | 0.781 | 0.469 | 0.594 |
| Lapa-12B alone | 0.7315 | 0.0000 | 0.7316 | 0.984 | 0.938 | 0.436 | 0.688 | **0.750** | 0.594 |

Aya and Aug-7 router are **not** new this week; they are the same-suite baselines. Solos and few-shot/cascade **are** new.

Repeats are essentially deterministic at T=0 (sd ≈ 0 on quality). Translate is the only bucket with a tiny sd (~0.002).

---

## 2. Deltas vs the few-shot router (the product we would ship)

Δ = few-shot router − other. Positive = router wins.

| vs | Δ overall | Main buckets |
|----|----------:|--------------|
| Aug-7 router (baseline prompt, old rules) | **+0.026** | alignment **+0.156**; everything else holds |
| Aya alone | **+0.057** | knowledge **+0.188**, alignment **+0.156**; chat −0.016 |
| Mamay-4B alone | **+0.081** | knowledge **+0.281**, translate +0.063, alignment +0.156; code −0.016 |
| Lapa alone | **+0.111** | translate **+0.384**, instruct +0.094, code +0.031; knowledge **tied** |
| Micro-cascade | **+0.000** | noise (translate 0.821 vs 0.820) |

**Read it:** routing beats every single model we measured on this suite. The +0.026 vs the old router is **almost entirely alignment few-shot**, not leakage.

---

## 3. Latency and GPU-seconds

| System | p50 ms | mean ms | GPU-s / prompt | GPU-s / 192-pass |
|--------|-------:|--------:|---------------:|-----------------:|
| **Router + few-shot** | **441** | 1593 | 1.59 | ~306 |
| Router + micro-cascade | 441 | 1594 | 1.59 | ~306 |
| Router matrix (Aug 7) | **425** | 1518 | ~1.52 | ~292 |
| Aya alone | 867 | 1310 | ~1.31 | ~252 |
| Mamay-4B alone | 1125 | 2155 | 2.16 | ~414 |
| Lapa alone | 1681 | 4628 | 4.63 | ~889 |

p50 ratio vs few-shot router: Aya **~2.0×**, Mamay-4B **~2.5×**, Lapa **~3.8×**.

Aya’s **mean** is a bit lower than the router’s because chat (slow, ~4 s) is a smaller share of Aya’s tail in the average, while the router still pays Mamay chat. The **median** is the fair “typical query” number — router wins that.

### Bucket p50 (ms), few-shot router vs solos

| Bucket | Router FS | Aya | Mamay-4B | Lapa |
|--------|----------:|----:|---------:|-----:|
| alignment | 189 | 1170 | 134 | 190 |
| knowledge | 252 | 1179 | 655 | 252 |
| code | 475 | 691 | 2476 | **15825** |
| translate | 518 | 519 | 1449 | 1728 |
| instruct | 706 | **289** | 699 | 938 |
| chat | 3927 | 3416 | 3823 | 5909 |

Lapa on **code** is the disaster cell for latency (~16 s p50). That is why a 12B solo is not “just a bit slower.”

---

## 4. Per-bucket: who actually wins on v4

| Bucket | Best single on v4 | Router uses | Router FS score | Note |
|--------|-------------------|-------------|----------------:|------|
| chat | Aya 1.000 | mamay4 | 0.984 | One Slack-too-short item; not a routing bug |
| instruct | tie 0.781 (Aya / Mamay / router) | mamay4 | 0.781 | Lapa 0.688 — do not send instruct to Lapa |
| knowledge | **Lapa 0.750** | lapa | **0.750** | Mamay 0.469 / Aya 0.562 — this is the routing win |
| alignment @ baseline | all 0.594 | lapa | 0.594 | prompt bug, not model |
| alignment @ few-shot | (slice: Mamay 0.781) | lapa + few-shot | **0.750** | matches Lapa few-shot cell |
| code | **Mamay 0.984** | qwen7 | 0.969 | Qwen slightly behind Mamay; still fine |
| translate | Aya / router **0.821** | aya | **0.821** | Lapa **0.436** — kills Lapa overall |

Route table is justified on **this** suite, not only on v3.

---

## 5. Traffic (rules)

| | mamay4 | lapa | qwen7 | aya |
|--|-------:|-----:|------:|----:|
| Aug 7 router (old rules) | 64 | 63 | 33 | 32 |
| Week 6 few-shot (fixed rules) | 63 | 65 | 32 | 32 |

Week-6 intents (rep 1): chat 31 · knowledge 33 · translate 32 · instruct 32 · code 32 · alignment 32.

Leakage fix moved a couple of items lapa-ward. **Quality on knowledge / instruct / code did not change** vs Aug 7. Soft leftover still exists (`chat-017` → knowledge).

---

## 6. Alignment (week 5 bake-off, confirmed on full suite this week)

Slice: 32 items (10 ethics + 22 social). Ethics quality **0.800** in every cell.

| Cell | Bucket | Social ref=1 correct | Social ref=1 → 2 |
|------|-------:|---------------------:|-----------------:|
| Lapa baseline | 0.594 | 0 / 11 | 11 / 11 |
| Lapa clarified | 0.750 | 8 / 11 | 2 / 11 |
| Lapa few-shot | 0.750 | **11 / 11** | 0 / 11 |
| Mamay-4B baseline | 0.594 | 1 / 11 | 8 / 11 |
| Mamay-4B few-shot | **0.781** | 9 / 11 | 1 / 11 |

On **social ref=1**, McNemar vs baseline is solid (Lapa few-shot p ≈ 0.001). On **all 32 items**, Δ is **not** significant (Lapa few-shot +0.156, CI crosses 0; fixes 11 / breaks 6 via 2→1 and 0→1).

**This week:** wiring Lapa few-shot into the full 192-item router moved alignment **0.594 → 0.750** and overall **0.816 → 0.842**, matching the slice.

**Do not** swap alignment to Mamay-4B as the product fix (both collapse at baseline). Mamay few-shot is a slightly better *cell*, but the shipped route stays Lapa + prompt.

---

## 7. Micro-cascade (S2)

Policy: first hop = matrix router; **social** items only; if answer is **`2`** or has no 0–2 digit → **one** retry on Mamay-4B with the same few-shot prompt. Ethics never escalated. No Mamay-12B.

| | Value |
|--|------:|
| Escalate rate | **3 / 192 = 1.56%** (all 3 repeats identical) |
| IDs | `ualign-social-2654`, `ualign-social-2724`, `ualign-social-3310` |
| Δ overall vs few-shot router | **−0.0002** (none) |
| Alignment bucket | still **0.750** |
| Extra GPU-s / pass | **+0.25** (~+0.08%) |

**Conclusion:** extra hop does not recover the remaining 2→1 few-shot errors. Drop cascade from the product story; keep it as a negative ablation.

---

## 8. Claims that are now allowed (and not)

**Allowed**

- On `mixed_ua_v4`, a rules router over four open models **beats Aya, Mamay-4B, and Lapa** on overall quality and on median latency.
- The knowledge win is **Lapa vs everyone else**; the translate win is **Aya vs Lapa**.
- The old alignment 0.594 was a **social 0/1/2 prompt collapse**, reproduced on both Lapa and Mamay-4B at baseline.
- Few-shot on social items is enough to land alignment **0.750** on the full suite.

**Not allowed**

- “Saves VRAM” — four hot models still reserve four GPUs.
- “Beats Mamay-12B / GPT / any closed model” — not measured on v4.
- “Works on 14k warehouse / production traffic” — not run.
- “Cascade / ensemble is how we get the last points” — cascade measured, Δ = 0.
- Treating 0.750 alignment as a solved 3-way scale — few-shot over-predicts **1**.

---

## 9. Artifacts

| What | Path |
|------|------|
| Mamay / Lapa mean±sd | `results/week_6_v4_solos/scores_mean_sd_v4_solos.json` |
| Compact comparison | `results/week_6_v4_solos/comparison_vs_aya_router.json` |
| Few-shot router | `results/week_6_router_fewshot/scores_mean_sd_router_fewshot.json` |
| Routing counts | `results/week_6_router_fewshot/routing_stats_rep1.json` |
| Cascade scores | `results/week_6_cascade_micro/scores_mean_sd_cascade_micro.json` |
| Escalate table | `results/week_6_cascade_micro/escalate_pct_table.json` |
| Raw JSONL | lab `~/Diploma/results/week_6_*/` |
| One-page claim | `docs/key_takeaways.md` §A |

**Slide sentence:** *Same-suite v4: rules router 0.842 / 441 ms vs Aya 0.785 / 867 ms, Mamay-4B 0.762 / 1125 ms, Lapa 0.732 / 1681 ms. +0.026 vs the old router is the alignment few-shot; a 1.6% cascade retry adds nothing.*
