#!/bin/bash
# mixed_ua_v6 screening + bake-off on ucu-lab-2240.
# Usage (from ~/Diploma):
#   bash scripts/week_v6_eval_pipeline.sh screen
#   bash scripts/week_v6_eval_pipeline.sh v6_solos
#   bash scripts/week_v6_eval_pipeline.sh v6_solo_mamay4   # one live model
#   bash scripts/week_v6_eval_pipeline.sh v6_solo_lapa
#   bash scripts/week_v6_eval_pipeline.sh v6_solo_aya
#   bash scripts/week_v6_eval_pipeline.sh v6_solo_mamay12  # S2 generalist
#   bash scripts/week_v6_eval_pipeline.sh v6_solo_qwen7    # Qwen as full generalist
#   bash scripts/week_v6_eval_pipeline.sh v6_rules
#   bash scripts/week_v6_eval_pipeline.sh v6_learned
set -euo pipefail
ROOT="${HOME}/Diploma"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

PHASE="${1:-screen}"
SCREEN="benchmarks/mixed_ua_v6_screen.jsonl"
V6="benchmarks/mixed_ua_v6.jsonl"
# IFEval constraints often need more than 256 tokens; pin it in run tags.
MAX_TOKENS="${MAX_TOKENS:-1024}"

wait_health() {
  local alias="$1"
  local port="$2"
  echo "waiting for ${alias} :${port} ..."
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "health 200 ${alias}"
      return 0
    fi
    sleep 15
  done
  echo "TIMEOUT waiting for ${alias} on :${port}" >&2
  return 1
}

case "$PHASE" in
  screen)
    if [[ ! -f "$SCREEN" ]]; then
      echo "missing $SCREEN — run: python scripts/sample_screening_suite.py" >&2
      exit 1
    fi
    wait_health mamay4 8003
    python scripts/run_small_router_cluster.py --mode mamay4 --bench "$SCREEN" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_screen
    wait_health lapa 8001
    python scripts/run_small_router_cluster.py --mode lapa --bench "$SCREEN" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_screen
    wait_health aya 8005
    python scripts/run_small_router_cluster.py --mode aya --bench "$SCREEN" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_screen
    if curl -sf "http://127.0.0.1:8002/v1/models" >/dev/null; then
      python scripts/run_small_router_cluster.py --mode mamay12 --bench "$SCREEN" --repeats 1 \
        --max-tokens "$MAX_TOKENS" --out-dir results/v6_screen
    fi
    python scripts/score_results.py results/v6_screen/*.jsonl --out results/v6_screen/scores_screen.json
    echo "next: PYTHONPATH=. python scripts/curate_discriminative_suite.py --suite $SCREEN --scores results/v6_screen/*_detail.jsonl --systems mamay4 lapa aya --out $V6 --keep-appendix"
    ;;
  v6_solo_mamay4)
    wait_health mamay4 8003
    python scripts/run_small_router_cluster.py --mode mamay4 --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    ;;
  v6_solo_lapa)
    wait_health lapa 8001
    python scripts/run_small_router_cluster.py --mode lapa --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    ;;
  v6_solo_aya)
    wait_health aya 8005
    python scripts/run_small_router_cluster.py --mode aya --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    ;;
  v6_solo_mamay12)
    wait_health mamay12 8002
    python scripts/run_small_router_cluster.py --mode mamay12 --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    ;;
  v6_solo_qwen7)
    wait_health qwen7 8004
    python scripts/run_small_router_cluster.py --mode qwen7 --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    ;;
  v6_solos)
    wait_health mamay4 8003
    python scripts/run_small_router_cluster.py --mode mamay4 --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    wait_health lapa 8001
    python scripts/run_small_router_cluster.py --mode lapa --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    wait_health aya 8005
    python scripts/run_small_router_cluster.py --mode aya --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_solos
    python scripts/score_results.py results/v6_solos/*.jsonl --out results/v6_solos/scores.json
    ;;
  v6_rules)
    wait_health mamay4 8003
    wait_health lapa 8001
    wait_health aya 8005
    python scripts/run_small_router_cluster.py --mode router_matrix --bench "$V6" --repeats 1 \
      --align-prompt fewshot --max-tokens "$MAX_TOKENS" --out-dir results/v6_rules
    python scripts/score_results.py results/v6_rules/*.jsonl --out results/v6_rules/scores.json
    ;;
  v6_learned)
    wait_health mamay4 8003
    wait_health lapa 8001
    wait_health aya 8005
    python scripts/run_small_router_cluster.py --mode router_knn --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_learned
    python scripts/run_small_router_cluster.py --mode router_clf --bench "$V6" --repeats 1 \
      --max-tokens "$MAX_TOKENS" --out-dir results/v6_learned
    python scripts/score_results.py results/v6_learned/*.jsonl --out results/v6_learned/scores.json
    python scripts/build_router_comparison.py --headline-only \
      --details results/v6_solos/scored/*_detail.jsonl \
               results/v6_rules/scored/*_detail.jsonl \
               results/v6_learned/scored/*_detail.jsonl \
      --lobo-report router/artifacts/train_report.json \
      --out results/v6_eval/router_comparison.json
    ;;
  *)
    echo "usage: $0 {screen|v6_solos|v6_solo_mamay4|v6_solo_lapa|v6_solo_aya|v6_solo_mamay12|v6_solo_qwen7|v6_rules|v6_learned}" >&2
    exit 1
    ;;
esac
