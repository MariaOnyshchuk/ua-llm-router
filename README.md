# UA Specialist Router (LiteLLM)

Short design: [DESIGN.md](DESIGN.md) · empirical summary: [docs/key_takeaways.md](docs/key_takeaways.md) · conclusion: [docs/conclusion.md](docs/conclusion.md)

## Quick start (routing rules only)

```bash
cd Diploma
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/smoke_test.sh
```

## Playground UI (model / pool / router)

Tunnel the lab vLLM ports, then:

```bash
uvicorn router.app:app --port 4010
# open http://localhost:4010/
```

The page talks to specialists **directly** on `:8001` Lapa, `:8003` Mamay-4B, `:8004` Qwen-7B, `:8005` Aya. You can force a single model, restrict the pool, switch rules v1/v2, compare specialists, try micro-cascade / ensemble vote, or run the composite planner and inspect every dependent step.

## Composite benchmark

Latest 36×3 result: oracle workflow **0.882**, rules-direct **0.812**,
hybrid planner **0.768**. See [docs/week_10_composite.md](docs/week_10_composite.md).

```bash
# CPU-only schema and unit checks
python scripts/build_composite_benchmark.py --check
python -m unittest discover -s tests -v

# Requires live specialist backends
python scripts/run_composite_router.py --limit 6 \
  --systems mamay4,router_direct,oracle,hybrid \
  --out-dir results/week_10_composite_smoke

python scripts/score_composite_results.py \
  results/week_10_composite_smoke/*.jsonl \
  --out results/week_10_composite_smoke/scores.json
```

## Full stack (OpenAI-compatible, when LiteLLM is up)

```bash
# Terminal A — LiteLLM
litellm --config litellm/config.yaml --port 4000

# Terminal B — rules router + playground
uvicorn router.app:app --port 4010

# Client
curl http://localhost:4010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role":"user","content":"Переклади: Hello world"}]
  }'
```

Aliases: `auto` | `lapa` | `mamay` | `gemma` | `qwen`
