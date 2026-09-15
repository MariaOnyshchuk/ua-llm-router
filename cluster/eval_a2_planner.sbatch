#!/bin/bash
#SBATCH --job-name=a2-planner
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/Diploma/logs/a2-planner-%j.out
#SBATCH --error=/home/%u/Diploma/logs/a2-planner-%j.err

# Planning-only A2 screen: current_v2 vs minimal_v3 on Mamay-4B then Mamay-12B.
# Sequential one-model-at-a-time (same GPU). No specialist execution.
#
#   sbatch cluster/eval_a2_planner.sbatch

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
# shellcheck disable=SC1091
set -a
[ -f "${ROOT}/.env" ] && . "${ROOT}/.env"
set +a
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${ROOT}/logs" "${ROOT}/results/a2_planner_grid" "${ROOT}/results/tables"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_USE_FLASHINFER_SAMPLER=0

serve_one() {
  local alias="$1"
  local model="$2"
  local port="$3"
  echo "[a2] host=$(hostname) serving ${alias} on :${port}"
  nvidia-smi -L
  vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    --enforce-eager &
  VLLM_PID=$!
  echo "[a2] waiting for ${alias} pid=${VLLM_PID}"
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[a2] health 200 ${alias} after ${i} checks"
      return 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[a2] vLLM exited before becoming healthy (${alias})" >&2
      return 1
    fi
    sleep 10
  done
  echo "[a2] TIMEOUT waiting for ${alias} on :${port}" >&2
  return 1
}

stop_serve() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    unset VLLM_PID
  fi
}
trap stop_serve EXIT

cd "${ROOT}"
COMBINED="${ROOT}/results/tables/a2_planner_grid.csv"
rm -f "${COMBINED}"
HEADER_WRITTEN=0

run_model() {
  local alias="$1"
  local out_csv="${ROOT}/results/tables/a2_planner_grid_${alias}.csv"
  PYTHONPATH=. python -u scripts/run_planner_grid.py \
    --bench benchmarks/mixed_ua_composite_v1.jsonl \
    --models "${alias}" \
    --out "${out_csv}" \
    --save-plans results/a2_planner_grid
  if [[ "${HEADER_WRITTEN}" -eq 0 ]]; then
    cat "${out_csv}" > "${COMBINED}"
    HEADER_WRITTEN=1
  else
    tail -n +2 "${out_csv}" >> "${COMBINED}"
  fi
}

serve_one mamay4 "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003
run_model mamay4
stop_serve
sleep 5

serve_one mamay12 "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0" 8002
run_model mamay12
stop_serve

echo "[a2] wrote ${COMBINED}"
