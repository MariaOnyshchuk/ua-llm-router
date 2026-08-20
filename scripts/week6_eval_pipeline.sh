#!/bin/bash
# Week 6 eval on ucu-lab-2240. Run after serving backends are healthy.
# Usage (from ~/Diploma):
#   bash scripts/week6_eval_pipeline.sh solos
#   bash scripts/week6_eval_pipeline.sh router
#   bash scripts/week6_eval_pipeline.sh cascade
set -euo pipefail
ROOT="${HOME}/Diploma"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

PHASE="${1:-solos}"
BENCH="benchmarks/mixed_ua_v4_balanced.jsonl"

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
  solos)
    wait_health mamay4 8003
    wait_health lapa 8001
    python scripts/run_small_router_cluster.py --mode mamay4 --bench "$BENCH" --repeats 3 \
      --out-dir results/week_6_v4_solos
    python scripts/run_small_router_cluster.py --mode lapa --bench "$BENCH" --repeats 3 \
      --out-dir results/week_6_v4_solos
    python scripts/score_repeat_dir.py results/week_6_v4_solos \
      --out results/week_6_v4_solos/scores_mean_sd_v4_solos.json
    ;;
  router)
    wait_health mamay4 8003
    wait_health lapa 8001
    wait_health qwen7 8004
    wait_health aya 8005
    python scripts/run_small_router_cluster.py --mode router_matrix --bench "$BENCH" --repeats 3 \
      --align-prompt fewshot --out-dir results/week_6_router_fewshot
    python scripts/score_repeat_dir.py results/week_6_router_fewshot \
      --out results/week_6_router_fewshot/scores_mean_sd_router_fewshot.json
    python scripts/summarize_router_jsonl.py results/week_6_router_fewshot/*_rep1_*.jsonl \
      --out results/week_6_router_fewshot/routing_stats_rep1.json
    ;;
  cascade)
    wait_health mamay4 8003
    wait_health lapa 8001
    wait_health qwen7 8004
    wait_health aya 8005
    python scripts/run_small_router_cluster.py --mode router_cascade_micro --bench "$BENCH" --repeats 3 \
      --align-prompt fewshot --out-dir results/week_6_cascade_micro
    python scripts/score_repeat_dir.py results/week_6_cascade_micro \
      --out results/week_6_cascade_micro/scores_mean_sd_cascade_micro.json
    python scripts/summarize_router_jsonl.py results/week_6_cascade_micro/*_rep1_*.jsonl \
      --out results/week_6_cascade_micro/escalate_stats_rep1.json
    python scripts/summarize_router_jsonl.py results/week_6_cascade_micro/*_rep*.jsonl \
      --out results/week_6_cascade_micro/escalate_stats_all_reps.json
    ;;
  *)
    echo "usage: $0 {solos|router|cascade}" >&2
    exit 1
    ;;
esac
