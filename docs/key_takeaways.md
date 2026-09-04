# Key takeaways — Ukrainian LLM router (diploma)

*Last updated: 2026-09-01. Weeks 7–10 pack (glossary, router diagrams, v4/v5/composite tables): [`docs/weeks_7_10_takeaways.md`](weeks_7_10_takeaways.md).*

## One-sentence claim

A **rules-based multi-lineage router** (Mamay-4B + Lapa + Aya + Qwen-Coder-7B) on `mixed_ua_v4` scores **0.816** at baseline and **0.842** with social few-shot — beating Aya (0.785), Mamay-4B (0.762), and Lapa (0.732) on the **same 192×3 suite**, with ~2–4× lower p50 than those singles. The old alignment hole (**0.594**) is a **social-scale prompt bug** (1→2); few-shot recovers the bucket to **0.750**. A micro-cascade (social `2` → one Mamay-4B retry) adds **1.6%** extra calls and **no quality**.

**Product stop:** rules v2 + few-shot = **0.848** (matches oracle). Extra hops and 4-bit do not beat it. Thesis write-up: [`docs/conclusion.md`](conclusion.md).

**Composite extension:** on 36 dependent tasks ×3, the stored workflow oracle
scores **0.882** vs one-hop rules **0.812**, proving a 0.070 composition
opportunity. The prompted hybrid planner reaches only **0.768** at 3.5 calls /
13.5 s p50. Correct workflows can help; general planning does not yet.
Details: [`docs/week_10_composite.md`](week_10_composite.md).

---

## Current router (intended)

| Bucket | Model | Port |
|--------|-------|------|
| code | Qwen2.5-Coder-7B (`qwen7`) — **optional**; v2 rules send code to Mamay-4B | 8004 |
| translate | Aya Expanse 8B (`aya`) | 8005 |
| knowledge | Lapa-12B (`lapa`) | 8001 |
| alignment | Lapa-12B (`lapa`) — **keep**; fix **prompt**, don’t swap model | 8001 |
| instruct | Mamay-4B (`mamay4`) | 8003 |
| chat | Aya Expanse 8B (`aya`) — v2 default | 8005 |

Code: `router/intent_rules.py`.

---

## Benchmarks

| Layer | Path | Size | Role |
|-------|------|-----:|------|
| **Eval suite (authoritative for claims)** | `benchmarks/mixed_ua_v4_balanced.jsonl` | 192 (32×6) | What we actually score |
| **Corpus warehouse** | `benchmarks/corpus/*.jsonl` | ≈14 644 | Source pool (ZNO, FLORES, UAlign, UA-Code) |
| Older suites | `mixed_ua_v1`…`v3` | smaller / unbalanced | Historical; week-4 bake-off used v3 |

Chat / instruct at scale barely exist in open UA data — those buckets stay mostly hand-written.

Full-corpus eval was deferred on purpose (time + imbalance + UA-Code unscored without Eolymp judge).

---

## Headline comparison

### A. Same suite — `mixed_ua_v4_balanced` (192×3, T=0)

| System | Overall | p50 | chat | code | translate | instruct | knowledge | alignment |
|--------|--------:|----:|-----:|-----:|----------:|---------:|----------:|----------:|
| **Rules v2 + few-shot** (chat→Aya, code→Mamay-4B) | **0.848** | 633 | **1.000** | **0.984** | 0.820 | 0.781 | **0.750** | **0.750** |
| Ensemble vote (align+ZNO, 1×) | 0.843 | 1201 | **1.000** | **0.984** | 0.822 | 0.781 | 0.688 | **0.781** |
| Rules v2 **4-bit bnb** (1×) | 0.830 | 1115 | **1.000** | **1.000** | 0.795 | 0.719 | 0.719 | **0.750** |
| **Oracle best + few-shot** | **0.848** | 633 | **1.000** | **0.984** | **0.821** | 0.781 | **0.750** | **0.750** |
| Router v1 + few-shot (code→Qwen) | 0.842 | **441** | 0.984 | 0.969 | **0.821** | 0.781 | **0.750** | **0.750** |
| Router + micro-cascade | 0.842 | 441 | 0.984 | 0.969 | 0.820 | 0.781 | 0.750 | 0.750 |
| Router (matrix, Aug 7) | 0.816 | **425** | 0.984 | **0.969** | 0.820 | 0.781 | **0.750** | 0.594 |
| Aya Expanse 8B alone | 0.785 | 867 | **1.000** | 0.953 | **0.821** | 0.781 | 0.562 | 0.594 |
| Mamay-4B alone | 0.762 | 1125 | 0.984 | **0.984** | 0.758 | 0.781 | 0.469 | 0.594 |
| Lapa-12B alone | 0.732 | 1681 | 0.984 | 0.938 | 0.436 | 0.688 | **0.750** | 0.594 |

- **Same-suite** Mamay-4B / Lapa solos are in `results/week_6_v4_solos/` (baseline prompts). Router **beats both** on overall quality **and** p50.
- Few-shot **rules** router vs Aug 7 baseline: **+0.026** overall; alignment **0.594 → 0.750**; traffic mamay4 63 · lapa 65 · qwen7 32 · aya 32.
- Gold-bucket oracle and **rules v2** (chat→Aya, code→Mamay-4B, no Qwen) both land **0.848 / 633 ms** on v4 — the regex table captures the oracle (only leftover: `chat-017`→knowledge). Details: [`docs/week_7_router_best.md`](week_7_router_best.md), [`docs/week_7_rules_v2.md`](week_7_rules_v2.md).
- Vs Aya: few-shot router **+0.057** overall, mainly knowledge + alignment; ~2× faster at p50.
- Lapa loses overall because **translate 0.436** (and slower code); knowledge/alignment at baseline match the old router.
- Micro-cascade (social `2`/no-digit → one Mamay-4B retry): **3/192 = 1.56%** escalate, **Δquality ≈ 0**, **+~0.25 GPU-s / pass**. Not worth the extra hop on this suite.
- Discrete ensemble (Lapa+Mamay-4B+Aya vote on alignment + ZNO): **56/192** voted, **3 flipped**, overall **0.848 → 0.843**, p50 **633 → 1201**, **+59 GPU-s**. Majority overwrote two correct Lapa ZNO answers; one social item improved. **Skip for the product.** Details: [`docs/week_7_ensemble.md`](week_7_ensemble.md).
- bitsandbytes 4-bit of the three specialists (same rules v2, same util 0.90): overall **0.848 → 0.830**, p50 **633 → 1115**, GPU-s **~295 → 473**. nvidia-smi still ~44 GB/GPU (KV reservation). **Keep bf16.** Details: [`docs/week_8_quant.md`](week_8_quant.md).

**HTTP:** 192/192 OK on week-6 runs.  
**Cost:** solos ~40 min wall (2 GPUs); few-shot router ~15 min (4 GPUs); cascade ~15 min.

Artifacts: `results/week_6_v4_solos/`, `results/week_6_router_fewshot/`, `results/week_6_cascade_micro/`, `results/week_7_router_best/`, `results/week_5_router_v4/`.  
**Full week-6 tables:** [`docs/week_6_takeaways.md`](week_6_takeaways.md).

### B. Context — single Mamay / Lapa on `mixed_ua_v3` (~182×3)

**Different suite** than v4 (item mix / counts differ). Use for specialist ranking and route design, **not** as “Router 0.816 vs Lapa 0.789” in one table cell.

| System | Overall (micro) | Macro | p50 ms | knowledge | alignment | code | translate |
|--------|----------------:|------:|-------:|----------:|----------:|-----:|----------:|
| Mamay-12B alone | **0.793** | 0.832 | 1188 | 0.594 | 0.750 | 0.967 | **0.809** |
| **Lapa-12B alone** | 0.789 | 0.831 | 1417 | **0.625** | **0.812** | 0.933 | 0.743 |
| Mamay-4B alone | 0.753 | 0.802 | 1101 | 0.469 | 0.708 | **0.983** | 0.778 |
| Old rules router (v3) | 0.750 | 0.800 | 1143 | 0.469 | 0.708 | 0.967 | 0.778 |
| Qwen2.5-Coder-3B alone | 0.700 | 0.768 | **410** | 0.422 | 0.653 | 0.967 | 0.689 |

Source: `results/week_4_specialist_bakeoff/scores_mean_sd_bakeoff.json`.

**How to read this vs the new router**

| Question | Answer from data we have |
|----------|--------------------------|
| Is the new router better than **Aya**? | **Yes** on v4 (baseline +0.031; few-shot +0.057, faster). |
| Is it better than **single Mamay-4B**? | **Yes** on v4: 0.842 / 0.816 vs **0.762**, p50 441/425 vs **1125 ms**. |
| Is it better than **single Lapa**? | **Yes** on v4 overall: 0.842 vs **0.732** (Lapa still wins knowledge at 0.750; dies on translate). Router alignment 0.59 was the **baseline prompt**. |
| Why route knowledge→Lapa, chat/instruct→Mamay4? | v3 bake-off: Lapa wins knowledge/alignment; Mamay4 wins code (among small models) and ties instruct; Mamay12 best translate then (before Aya). |

### C. Per-bucket intuition (cross-suite, provisional)

| Bucket | Best single (v3) | What v4 router uses | v4 router score |
|--------|------------------|---------------------|----------------:|
| chat | Mamay4 / Lapa / Mamay12 (all ~1.0) | mamay4 | 0.984 |
| instruct | tie ~0.875 | mamay4 | 0.781 (harder v4 items) |
| knowledge | **Lapa 0.625** | lapa | **0.750** |
| alignment | **Lapa 0.812** (v3) | lapa + **few-shot** (week 6) | **0.750** (was 0.594 at baseline; see §E) |
| code | Mamay4 0.983 (v3) | qwen7 | 0.969 |
| translate | Mamay12 0.809 (v3) | aya | 0.820 |

Gap that matters for the thesis: **alignment prompt** — not which model owns the bucket (see §E).

### D. Latency & resource utilisation

Split two claims: **per-request latency / active GPU work** vs **resident hardware when all specialists stay loaded**.

#### Latency

| System | Suite | p50 | mean |
|--------|-------|----:|-----:|
| **Router (matrix / few-shot)** | v4 | **~425–441 ms** | ~1.6 s |
| Aya alone | v4 | ~867 ms | ~1.3 s |
| Mamay-4B alone | v4 | ~1125 ms | ~2.2 s |
| Lapa alone | v4 | ~1681 ms | ~4.6 s |
| Mamay-12B alone | v3 | ~1.2 s | ~1.7 s |

- Same-suite: router **~2× lower p50** than Aya, **~2.5×** vs Mamay-4B, **~3.8×** vs Lapa.
- Router **p50 ≪ mean** because short buckets dominate the median; **chat** is slow (~4.1 s p50) and pulls the average up.

Router bucket p50 (v4): alignment ~191 · knowledge ~253 · code ~476 · translate ~518 · instruct ~711 · chat ~4095 ms.

Active compute: **~1.5 GPU-s / prompt** · ~5 min / 192-pass · ~15 min for 3×192.

#### Resources / utilisation

| Lens | Router (how we ran the v4 job) | Single Mamay / Lapa |
|------|--------------------------------|--------------------|
| Calls per query | **1** specialist | 1 |
| Active GPU work | ~1 GPU sequential | ~1 GPU |
| Resident VRAM | **4 models hot** ≈ **44 GB × 4 ≈ 174 GB** reserved (`gpu_memory_util≈0.9`; includes KV-cache reservation, not bare weights) | 1 model on 1 GPU |
| Wall eval cost | cheap in time | Lapa especially slow on mean latency |

**Honest thesis line:** the matrix router improves **response time and active GPU-seconds**, not “uses less VRAM when fully deployed.” An always-on four-model serve still occupies four GPUs of reserved memory. Fair weight-VRAM comparisons need **one model live at a time** (see `resource_protocol` in run metas).

| Serving style | Latency | Quality upside | Hardware |
|---------------|---------|----------------|----------|
| Single 12B (Lapa / Mamay12) | slower | strong alone | 1 GPU |
| Rules router (4 loaded) | fast median | specialists | 4 GPUs resident |
| Selective cascade | +cost only on escalate | recover hard cases | 1–2 calls |
| Ensemble always-on | slowest / dearest | upper bound (S3) | N calls × N GPUs |

Cost snapshot (lab): electricity negligible (~0.12 kWh for router 3×192); cloud-equivalent if billed ~$0.4–1.2 for resident 4×GPU-hours of that job. Details: `results/week_5_router_v4/cost_estimate_v4.json`.

**Slide sentence:** *The matrix router cuts median latency roughly in half vs Aya and ~2–3× vs Mamay/Lapa-class singles, at ~1.5 GPU-s per query — but a always-on four-model deploy still occupies four GPUs of reserved memory.*

### E. Alignment bake-off (finished) — prompt problem, not model

**Verified** against raw JSONL + `results/week_5_alignment_bakeoff/scores_alignment_variants.json` (2 deterministic reps each; ethics subset unchanged).

Slice: 32 alignment items from `mixed_ua_v4` (10 ethics + 22 social). Variants rewrite **social** prompts only.

| Cell | Bucket quality | Ethics | Social ref=1 correct | Social ref=1 → answered 2 |
|------|---------------:|-------:|---------------------:|--------------------------:|
| Lapa @ baseline | 0.594 | 0.800 | **0 / 11** | **11 / 11** |
| Lapa @ clarified | **0.750** | 0.800 | 8 / 11 | 2 / 11 |
| Lapa @ few-shot | **0.750** | 0.800 | **11 / 11** | 0 / 11 |
| Mamay-4B @ baseline | 0.594 | 0.800 | 1 / 11 | 8 / 11 |
| Mamay-4B @ few-shot | **0.781** | 0.800 | 9 / 11 | 1 / 11 |

**Trust checks (pass):**
- Both repeats **byte-identical** scores per cell (`deterministic: true`).
- Ethics quality **pinned at 0.800** in all five cells → movement is from the social prompt, not scorer drift.
- Router alignment **0.5938** matches Lapa @ baseline exactly → the router hole is this prompt bug.

**Mechanism vs bucket-wide significance:**
- On **social ref=1**, McNemar vs baseline: Lapa few-shot **p ≈ 0.001**; Lapa clarified / Mamay few-shot **p ≈ 0.008** — the 1→2 collapse fix is solid.
- On **all 32 alignment items**, bucket Δ is **not** significant: Lapa few-shot +0.156 CI [−0.094, +0.406]; Mamay few-shot +0.188 CI [0.000, +0.375].
- Tradeoff: variants push toward predicting **1**, trading 1→2 for new **2→1** (and some 0→1). Lapa few-shot **fixes 11 / breaks 6**; clarified **fixes 8 / breaks 3** — both land at 0.75.

**Implication:** swapping alignment Lapa ↔ Mamay-4B does **not** fix the hole (both collapse at baseline). Few-shot is now on the eval path (`--align-prompt fewshot`); rules v2 lands at alignment **0.750** / overall **0.848**. Micro-cascade on leftover social `2`s did not move the bucket.

**1b — clearer 2-class contrast: skipped.** After few-shot the 1→2 collapse is gone (11/11). The leftover social errors are the opposite: **4× `2→1`** plus **2× `0→1`** (over-predicting the everyday class). Ethics is a separate 0.800 ceiling (prompt variants never rewrite those 10 items). A contrastive 1-vs-2 pair might win a point or two on the same 32, but the bucket McNemar is already non-significant, and full-suite rules v2 already **matches the oracle**. Not worth another GPU pass; product prompt stays few-shot.

Artifacts: `scripts/score_alignment_variants.py`, `scripts/alignment_prompt_variants.py`, `results/week_5_alignment_bakeoff/scores_alignment_variants{,_flat,_detail}.*`. Older `scores_alignment_bakeoff.json` is baseline-only / superseded for variant claims.

---

## Bugs found (must fix before new combo tables)

### 1. Router bucket leakage — **fixed in code**

Previously:
- Instruct items (`if-008/013/021`…) leaked to **knowledge** (`що таке`, `What is`, year `2026`…)
- Hand-written `know-002`…`008` fell through to **chat**
- Bare `тільки` also stole ZNO/code if instruct was naively prioritized

Fix (`router/intent_rules.py`):
- Strong instruct markers **before** knowledge/code (not bare `тільки`)
- UA factoid openers for knowledge (`У якому`, `Яка/Яке…`, `Скільки`, `Хто автор`…)

Check: `python scripts/check_router_leakage.py` → 11/11 hard cases OK. Soft leftover: `chat-017` → knowledge.

**Re-run** on full v4 with this fix: week-6 few-shot router (`results/week_6_router_fewshot/`). Traffic vs Aug 7: mamay4 64→63, lapa 63→65, qwen7 33→32 (aya 32). Knowledge/instruct/code scores unchanged vs baseline router — the +0.026 is from the alignment prompt, not leakage.

### 2. Alignment social-scale collapse — **resolved as prompt bug**

See **§E**. Baseline Lapa (and Mamay-4B) map social “1 — очікувано” → **2**; few-shot / clarified recover. **Do not** treat “move alignment to Mamay-4B” as the fix.

---

## Experiment roadmap (agreed direction)

| Priority | Idea | Verdict |
|----------|------|---------|
| 1 | Alignment ablation | **Done** — prompt fix, not route swap |
| 1b | Better social prompt (keep 1-class few-shot, clearer 2-class contrast) | **Skipped** — few-shot already 0.750 (11/11 on social-1); leftover is 2→1 overcorrection; overall already = oracle 0.848 |
| 2 | Wire improved alignment prompt into eval/router path; re-run router v4 (+ leakage fix) | **Done** — `--align-prompt fewshot` → 0.842 / alignment 0.750; rules v2 **0.848** |
| 3 | **Selective cascade** (social 2 → Mamay-4B retry, no Mamay-12) | **Done** — 1.56% escalate, Δquality ≈ 0 |
| 4 | **Ensemble / multi-agent** on discrete labels | **Done** — 0.843 vs 0.848; 3 flips net −1; skip product |
| 5 | **Quantize** Mamay-4B / Lapa / Aya (bitsandbytes 4-bit) | **Done** — 0.830 / slower; keep bf16 |
| 5b | Pack specialists on **one GPU** | **Not measured** — nvidia-smi ~44 GB/card at `gpu_memory_util=0.90` (KV fills the card). Needs lower util or two vLLMs colocated |
| — | Full Mamay-4B / Lapa solo on v4 192×3 | **Done** (`week_6_v4_solos`) |
| **6** | **S2: Mamay-12B on v4** — does composition beat scaling? | **Next** — ~40 min; [`docs/week_9_baselines.md`](week_9_baselines.md) |
| **7** | **S3: hosted frontier API on v4** — how far is the unavailable option? | **Next** — ~$1–4; `scripts/run_api_baseline.py` |
| **8** | Composite benchmark + hybrid orchestrator | **Done** — oracle 0.882, one-hop 0.812, hybrid 0.768; use fixed workflows, not general planner |
| — | Full ~14k corpus ×3 | Optional later; ~18 h router wall estimate |

Cascade should be **narrow** (e.g. instruct format fail→retry), not generic weak confidence that previously made `router_cascade ≈ router_small` on v3.

---

## Lab ops (reminder)

- Slurm lives on **`ucu-lab-2240`**, not the laptop: `ssh ucu-lab-2240`
- Check queue / GPUs: `squeue`, `sinfo -p gpu`, `nvidia-smi`
- Health: `curl …:8003/v1/models` (mamay4), `:8001` (lapa), `:8004` (qwen7), `:8005` (aya)
- Partition is **1 node × 4× RTX 6000 Ada** — when another user holds all four, serves stay `PD`

---

## Files to open first

| What | Where |
|------|--------|
| Route rules | `router/intent_rules.py` |
| Leakage check | `scripts/check_router_leakage.py` |
| Alignment variants (scored) | `results/week_5_alignment_bakeoff/scores_alignment_variants.json` |
| Alignment prompt helpers | `scripts/alignment_prompt_variants.py`, `scripts/score_alignment_variants.py` |
| Router scores (baseline) | `results/week_5_router_v4/scores_mean_sd_router_v4.json` |
| Weeks 7–10 (oracle / router / suites / metrics) | [`docs/weeks_7_10_takeaways.md`](weeks_7_10_takeaways.md) |
| Problem statement / scope | [`docs/thesis_scope.md`](thesis_scope.md) |
| Missing baselines plan | [`docs/week_9_baselines.md`](week_9_baselines.md) |
| Thesis conclusion | [`docs/conclusion.md`](conclusion.md) |
| Week-6 tables (solos / few-shot / cascade) | `docs/week_6_takeaways.md` |
| Router few-shot / solos / cascade scores | `results/week_6_*` |
| Routing / success / cost | `week_5_router_v4/routing_stats_rep1.json`, `week_6_router_fewshot/routing_stats_rep1.json` |
| Results index | `results/README.md` |
| Older route doc (partly stale on Aya/Qwen) | `docs/router_matrix.md` |
