# Local error analysis (no new GPU runs)

*Generated from existing JSONL · 2026-08-18*

Sources:
- `week_5_router_v4/router_matrix_*_rep1_*.jsonl` + scored detail (routing = **pre-leakage-fix**)
- `week_5_alignment_bakeoff/scores_alignment_variants.json`
- Bucket means: router vs Aya rescored aggregates

Also: interactive canvas `error-analysis-local.canvas.tsx`.

---

## 1. Misroutes before the fix (13 / 192)

| Flow | n | Model | IDs / cause |
|------|--:|-------|-------------|
| knowledge → chat | 6 | mamay4 | know-002…008 — no factoid opener in old regex |
| instruct → knowledge | 3 | lapa | if-008/013/021 — `What is` / `Що таке` / year `2026` |
| chat → knowledge | 2 | lapa | chat-004 `факт`; chat-017 `що таке` |
| instruct → chat | 1 | mamay4 | if-020 (still Mamay; wrong intent label) |
| instruct → code | 1 | qwen7 | if-032 python fence → **score 0** |

**Current rules:** fix **12/13**; leftover **chat-017** → knowledge.

**Score irony:** mean on misrouted items **0.92** vs **0.81** on correctly routed — hand-written know-* scored 1.0 on Mamay-4B, while Lapa on the rest of knowledge averaged ~0.69. Leakage hurt *validity* more than headline quality (except if-032).

| Gold bucket | Mis n | Mean if mis | Mean if OK |
|-------------|------:|------------:|-----------:|
| knowledge | 6 | 1.000 | 0.692 |
| instruct | 5 | 0.800 | 0.778 |
| chat | 2 | 1.000 | 0.983 |

---

## 2. Alignment confusion (social)

### Lapa @ baseline (bucket 0.594)
| ref\pred | →0 | →1 | →2 |
|----------|---:|---:|---:|
| 0 | 6 | 0 | 0 |
| **1** | 0 | **0** | **11** |
| 2 | 0 | 0 | 5 |

### Lapa @ few-shot (bucket 0.750)
| ref\pred | →0 | →1 | →2 |
|----------|---:|---:|---:|
| 0 | 4 | 2 | 0 |
| **1** | 0 | **11** | **0** |
| 2 | 0 | **4** | 1 |

### Mamay-4B @ few-shot (bucket 0.781)
| ref\pred | →0 | →1 | →2 |
|----------|---:|---:|---:|
| 0 | 6 | 0 | 0 |
| 1 | 1 | 9 | 1 |
| 2 | 0 | 3 | 2 |

Ethics fixed across cells at **0.800** (3×0→0, 2×0→1, 5×1→1).

---

## 3. Router vs Aya — who wins which bucket

| Bucket | Router | Aya | Δ | Winner |
|--------|-------:|----:|--:|--------|
| chat | 0.984 | 1.000 | −0.016 | **Aya** |
| translate | 0.820 | 0.821 | −0.001 | Aya ≈ |
| instruct | 0.781 | 0.781 | 0 | Tie |
| alignment | 0.594 | 0.594 | 0 | Tie |
| code | 0.969 | 0.953 | +0.016 | Router |
| knowledge | 0.750 | 0.562 | **+0.188** | **Router** |

### Chat (why Aya wins)
Only **one** imperfect router chat item: **chat-019** — Slack release congrats. Mamay-4B: `Чудовий реліз! 🎉` → score **0.5** (`length_ok=false`). Correctly routed to mamay4/chat. Not a routing bug — scorer length / brevity.

### Other router zeros (for context)
- **Instruct (7):** format failures on mamay4 (yaml fences, word count, etc.) + if-032 misroute
- **Knowledge (8):** know-001 + 7 ZNO wrong letters on Lapa
- **Alignment (13):** 2 ethics + 11 social 1→2 (baseline prompt)
- **Code (1):** code-004 on qwen7
- **Translate:** continuous metric — almost all &lt;1.0 by design

---

## Takeaways
1. Fix leakage for **honest buckets**, not because misroutes tanked the mean.
2. Alignment story is prompt/confusion-matrix, not “Lapa bad.”
3. vs Aya: thesis win = **knowledge** (+ latency); chat gap is one short Slack message.
