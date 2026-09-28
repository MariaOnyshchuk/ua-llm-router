# Conclusion (thesis draft)

*Empirical line, `mixed_ua_v4_balanced` 192 items, T=0, seed=42, max_tokens=256.*

## Claim

A **rules-based multi-lineage router** over three open specialists — Mamay-4B (instruct / UA code), Lapa-12B (knowledge / alignment), Aya Expanse 8B (translate / default chat) — with a **few-shot fix on social UAlign prompts**, matches the gold-bucket oracle on this suite:

**rules v2 + few-shot = 0.848 overall, p50 633 ms.**

That is better quality **and** lower median latency than any single specialist **in the pool** on the same items (Aya 0.785 / 867 ms, Mamay-4B 0.762 / 1125 ms, Lapa 0.732 / 1681 ms). It is not better quality than one larger open model: Mamay-12B alone scores **0.874** / p50 1352 ms. Qwen-Coder-7B is not in the product table: on v4, Mamay-4B wins code (0.984 vs 0.969).

This is **routing across independently trained models**, not MoE inside one network. “Разом ≈ великий модель” means complementary coverage, not summed parameters.

## What moved the number

| Lever | Effect on product |
|-------|-------------------|
| Regex intent table (leakage-fixed) | Captures the oracle map; leftover `chat-017` → knowledge |
| Chat → Aya, code → Mamay-4B (rules v2) | 0.842 → **0.848**; matches oracle |
| Social few-shot on alignment | Alignment 0.594 → **0.750**; the hole was a 1→2 prompt bug, not the wrong specialist |
| Micro-cascade (social `2` → Mamay-4B retry) | 1.56% extra calls, **Δquality ≈ 0** |
| Ensemble majority (alignment + ZNO) | **0.843** vs 0.848; 3 flips net −1; p50 633 → 1201 |
| bitsandbytes 4-bit (same util 0.90) | **0.830**, slower (p50 1115); nvidia-smi still ~44 GB/GPU |

**Product:** rules v2 + few-shot, bf16, one hop. Extra hops and 4-bit do not beat 0.848.

## Alignment (priority 1 / 1b)

Baseline Lapa and Mamay-4B both map social “1 — очікувано” → **2**. Few-shot recovers **11/11** of those items; ethics stays pinned at 0.800 (control). Swapping the alignment owner does not fix the baseline collapse.

A further “clearer 2-class contrast” pass on the same 32 items was **not run**. After few-shot the remaining social mistakes are over-predicting 1 (`2→1`, `0→1`), the full 32-item Δ is not significant, and the router already matches the oracle overall.

## Resources (honest)

The router improves **per-query latency and active GPU-seconds** (~1.5 GPU-s / prompt), not resident memory of the bf16 product. An always-on three-model serve at `gpu_memory_utilization=0.90` still occupies **three GPUs**. Week-8 4-bit did not change that picture. A later FP8 colocation does: the same three specialists fit on one card (45 668 MiB) at **0.842** / p50 723 ms, with knowledge down from 0.750 to 0.688. Mamay-12B alone is one card at the 0.90 reservation and scores higher overall (**0.874**) at p50 1352 ms and ~2.5 GPU-s / prompt.

## Limitations

- Authoritative claims use **192 items × 3 repeats** (`mixed_ua_v4`), not the ~14k corpus warehouse.
- Chat and instruct at scale barely exist in open UA data; those buckets stay mostly hand-written.
- `mixed_ua_v5` (+32 HumanEval) is a harder, **different** suite (code 0.789 mix; chat hurt by global max_tokens=512). Do not compare 0.796 to 0.848 as a router regression.
- No ~70–120B monolingual UA reference was served. On this suite the open-model comparison above the pool is Mamay-12B at 0.874, not the pool oracle.

## Composite-task extension

The 0.848 result concerns a heterogeneous set of **single-skill** requests. A
second benchmark now tests 36 requests with dependent workflows:
translate→code, knowledge→explain, and translate→knowledge→write. It compares a
direct model, the old one-hop router, a stored workflow oracle, and a validated
Mamay-4B JSON planner.

The oracle workflow reaches **0.882**, above the one-hop router at **0.812**:
there is measurable composition headroom. The hybrid planner reaches only
**0.768** despite 0.991 valid plans and 0.806 exact workflow match, while using
3.5 calls and 13.5 s p50. The result is therefore not “multi-agent is always
better.” Correct fixed workflows help the three-stage source→JSON family, while
unnecessary decomposition hurts tasks a single model already solves.

A follow-on composite v2 suite (200 items, train/dev/test, five families) shows
that a **constrained template selector** matches oracle workflows exactly on the
held-out test split (exact match **1.000**). Distilling a planner is not
justified on this bounded surface; the remaining open measurement is live
specialist execution on that test split (lab sbatch), not further prompt tuning.

## One paragraph for the diploma

На змішаному українському наборі `mixed_ua_v4` (192 запити) правила маршрутизації v2 з few-shot для соціальної шкали UAlign дають **0.848** якості при медіанній затримці 633 мс — стільки ж, скільки оракул «найкращий спеціаліст на кожен бакет», і більше, ніж будь-яка одна модель пулу. Одна більша відкрита модель, Mamay-12B, на тому самому протоколі дає **0.874** при 1352 мс: композиція не обганяє масштабування за якістю, але лишається швидшою і кращою на knowledge, code і translate. Додаткові стрибки (мікро-каскад, голосування ансамблю) і 4-bit bitsandbytes точку 0.848 **не покращують**. Ті самі три спеціалісти в online FP8 вміщаються на одну карту (45 668 МіБ) з якістю **0.842** і p50 723 мс; просідання в knowledge (0.750 → 0.688).
