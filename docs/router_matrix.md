# Matrix router — multi-lineage model pool

Built from `results/week_4_specialist_bakeoff/specialist_matrix_bakeoff.json`
(margin ≥ 0.02 vs Mamay-4B).

## Route table (implemented in `router/intent_rules.py`)

| Bucket | Model | Port | Why |
|--------|-------|------|-----|
| **code** | Qwen-Coder-7B | 8004 | Larger code specialist; bake-off pending |
| **translate** | Aya Expanse 8B | 8005 | Cohere multilingual lineage; bake-off pending |
| **knowledge** | Lapa-12B | 8001 | +0.156 vs Mamay-4B (also > Mamay-12B) |
| **alignment** | Lapa-12B | 8001 | +0.104 vs Mamay-4B (also > Mamay-12B) |
| **instruct** | Mamay-4B | 8003 | Tie |
| **chat** (default) | Mamay-4B | 8003 | Default |

This is a **real multi-model router** with four targets across three lineages:
Gemma-UA (Mamay/Lapa), Cohere (Aya), and Qwen. Aya/Qwen7 routes remain
provisional until their v4 bake-offs finish.

## Run on lab

```bash
# start the four backends
sbatch cluster/serve_lapa.sbatch      # :8001
sbatch cluster/serve_mamay4.sbatch     # :8003
sbatch cluster/serve_qwen7b.sbatch     # :8004
sbatch cluster/serve_aya.sbatch        # :8005

# matrix router on balanced suite
PYTHONPATH=. .venv/bin/python scripts/run_small_router_cluster.py router_matrix \
  --bench benchmarks/mixed_ua_v4_balanced.jsonl --repeats 3
```

Compare against single models on the same suite: `mamay4`, `lapa`, `aya`, `qwen7`.

## Efficiency honesty

Knowledge / alignment / translate winners are **12B**. Quality can beat Mamay-4B alone;
VRAM vs one Mamay-12B is **not** won if all three stay hot. Report peak VRAM with
one-model-hot protocol, or concurrent footprint if claiming multi-serve deployment.
