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
| **`mixed_ua_v6.jsonl`** | **2414** | 683×3 + instruct 327; chat/code appendix | **Eval freeze** — [`docs/mixed_ua_v6.md`](../docs/mixed_ua_v6.md) |

## Tiny samples (`samples/`) — legacy / hand extras only

Historical 24–48 item draws + hand-written chat/instruct/code extras. Prefer `corpus/` for anything new.

## Coding

- **Full UA-Code-Bench problems:** `corpus/uacode_problems_full.jsonl` (468).
- **Scored code in v4:** still hand-written `code-001`…`032` (exec tests). UA-Code needs Eolymp judge for real pass/fail.

