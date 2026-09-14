#!/bin/bash
#SBATCH --job-name=v6-solo
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/Diploma/logs/v6-solo-%x-%j.out
#SBATCH --error=/home/%u/Diploma/logs/v6-solo-%x-%j.err

# Co-located serve + frozen mixed_ua_v6 solo (1 pass, T=0, seed=42, max_tokens=1024).
# Full clean re-run (not a backfill). Fail-fast runner stops if vLLM dies.
#
#   sbatch --job-name=v6-mamay12 --export=ALL,ALIAS=mamay12 cluster/eval_v6_solo.sbatch
#   sbatch --job-name=v6-qwen7  --export=ALL,ALIAS=qwen7  cluster/eval_v6_solo.sbatch

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
mkdir -p "${ROOT}/logs" "${ROOT}/results/v6_solos"

ALIAS="${ALIAS:-mamay12}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_USE_FLASHINFER_SAMPLER=0

case "${ALIAS}" in
  mamay12)
    MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0}"
    PORT="${PORT:-8002}"
    PHASE="v6_solo_mamay12"
    ;;
  qwen7)
    MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    PORT="${PORT:-8004}"
    PHASE="v6_solo_qwen7"
    ;;
  *)
    echo "unknown ALIAS=${ALIAS} (expected mamay12|qwen7)" >&2
    exit 1
    ;;
esac

echo "[v6-solo] host=$(hostname) alias=${ALIAS} port=${PORT} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

vllm serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --enforce-eager &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[v6-solo] waiting for vLLM pid=${VLLM_PID} on :${PORT}"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[v6-solo] health 200 after ${i} checks"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[v6-solo] vLLM exited before becoming healthy" >&2
    exit 1
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "[v6-solo] TIMEOUT waiting for ${ALIAS} on :${PORT}" >&2
  exit 1
fi

cd "${ROOT}"
bash scripts/week_v6_eval_pipeline.sh "${PHASE}"
echo "[v6-solo] finished ${PHASE}"
