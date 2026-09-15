#!/bin/bash
#SBATCH --job-name=a2-ac-t512
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/Diploma/logs/a2-agentcoma-t512-%j.out
#SBATCH --error=/home/%u/Diploma/logs/a2-agentcoma-t512-%j.err

# Token-budget probe: Mamay-4B current_v2 vs minimal_v3, max_tokens=512,
# only the longest AgentComA families (travel_agent + science_experiments).
# Planning only. Do not overwrite the 256-token 50-item artifacts.
#
#   sbatch cluster/eval_a2_agentcoma_t512_long.sbatch

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
mkdir -p "${ROOT}/logs" "${ROOT}/results/a2_agentcoma_t512_long" "${ROOT}/results/tables"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[a2-t512] host=$(hostname) serving mamay4 on :8003"
nvidia-smi -L
vllm serve "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" \
  --host 127.0.0.1 \
  --port 8003 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --enforce-eager &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[a2-t512] waiting for mamay4 pid=${VLLM_PID}"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:8003/v1/models" >/dev/null; then
    echo "[a2-t512] health 200 after ${i} checks"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[a2-t512] vLLM exited before becoming healthy" >&2
    exit 1
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:8003/v1/models" >/dev/null; then
  echo "[a2-t512] TIMEOUT waiting for mamay4" >&2
  exit 1
fi

cd "${ROOT}"
PYTHONPATH=. python -u scripts/run_planner_grid.py \
  --bench benchmarks/agentcoma_uk_50_composite.jsonl \
  --family travel_agent,science_experiments \
  --models mamay4 \
  --max-tokens 512 \
  --timeout 300 \
  --out results/tables/a2_agentcoma_mamay4_t512_long.csv \
  --save-plans results/a2_agentcoma_t512_long

echo "[a2-t512] wrote results/tables/a2_agentcoma_mamay4_t512_long.csv"
