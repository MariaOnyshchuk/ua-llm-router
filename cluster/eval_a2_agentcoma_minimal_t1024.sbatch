#!/bin/bash
#SBATCH --job-name=a2-ac-v3-1k
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/Diploma/logs/a2-agentcoma-minimal-t1024-%j.out
#SBATCH --error=/home/%u/Diploma/logs/a2-agentcoma-minimal-t1024-%j.err

# Mamay-4B + minimal_v3 on all 50 AgentComA-UK items.
# Planning only, max_tokens=1024. Raw first and repair responses are retained.

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
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p \
  "${ROOT}/logs" \
  "${ROOT}/results/a2_agentcoma_minimal_t1024" \
  "${ROOT}/results/tables"

echo "[a2-v3-1k] host=$(hostname) serving mamay4 on :8003"
nvidia-smi -L
vllm serve "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" \
  --host 127.0.0.1 \
  --port 8003 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --trust-remote-code \
  --enforce-eager &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:8003/v1/models" >/dev/null; then
    echo "[a2-v3-1k] health 200 after ${i} checks"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[a2-v3-1k] vLLM exited before becoming healthy" >&2
    exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:8003/v1/models" >/dev/null

cd "${ROOT}"
PYTHONPATH=. python -u scripts/run_planner_grid.py \
  --bench benchmarks/agentcoma_uk_50_composite.jsonl \
  --models mamay4 \
  --variants minimal_v3 \
  --max-tokens 1024 \
  --timeout 600 \
  --out results/tables/a2_agentcoma_mamay4_minimal_t1024.csv \
  --save-plans results/a2_agentcoma_minimal_t1024

echo "[a2-v3-1k] complete"
