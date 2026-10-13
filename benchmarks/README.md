# Benchmarks inventory

## Full corpora (thousands) — `corpus/`

| File | n |
|------|--:|
| `zno_knowledge_full.jsonl` | 4 776 |
| `flores_translate_full.jsonl` | 4 018 |
| `ualign_ethics_full.jsonl` + `ualign_social_full.jsonl` | 5 382 |
| `uacode_problems_full.jsonl` | 468 unique problems |
| **Total extracted** | **≈14 644** |

```bash
python scripts/extract_full_corpora.py
python scripts/extract_full_corpora.py --only ifeval,belebele,mmlu,arc,wmt22
```

See `corpus/README.md`.

## Suites (do not overwrite old ones — keep for historical scores)

| File | n | Balance | Use |
|------|--:|---------|-----|
| `mixed_ua_v0.jsonl` | 40 | 8×5 | Smoke / early wiring |
| `mixed_ua_v1.jsonl` | 64 | +ZNO | Historical week 2 |
| `mixed_ua_v2.jsonl` | 160 | messy | Historical week 2–3 |
| `mixed_ua_v3.jsonl` | 182 | messy (tr56/al48/…) | Historical week 3–4 bake-off |
| **`mixed_ua_v4_balanced.jsonl`** | **192** | **32×6** | Historical eval (weeks 6–10) |
| `mixed_ua_v6_screen.jsonl` | 4237 | warehouse draw | Discriminability screening (not for claims) |
| `mixed_ua_v6.jsonl` | 2414 | 683×3 + instruct 327; chat/code appendix | Withdrawn 12 Oct 2026 (inverted selection) — [`docs/mixed_ua_v6.md`](../docs/mixed_ua_v6.md) |
| **`mixed_ua_v7.jsonl`** | **4237** | whole screening pool plus HumanEval for code; dev 1542, test 2859 (knowledge 794, translate 795, alignment 800, instruct 361, code 109) | **Claim suite.** No score-based selection; v4 items forced to dev. Build: `scripts/benchmarks/build_mixed_ua_v7.py` |
| `intent_natural_v1.jsonl` | 72 | 12 hand-written everyday requests per intent | Routing diagnostics only (right intent, right model). Not a quality benchmark |

## Tiny samples (`samples/`) — legacy / hand extras only

Historical 24–48 item draws + hand-written chat/instruct/code extras. Prefer `corpus/` for anything new.

## Coding

- **Full UA-Code-Bench problems:** `corpus/uacode_problems_full.jsonl` (468).
- **Scored code in v4:** still hand-written `code-001`…`032` (exec tests). UA-Code needs Eolymp judge for real pass/fail.
