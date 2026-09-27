#!/bin/bash
#SBATCH --job-name=s2-mamay12-v4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/Diploma/logs/s2-mamay12-v4-%j.out
#SBATCH --error=/home/%u/Diploma/logs/s2-mamay12-v4-%j.err

# S2: Mamay-12B alone on mixed_ua_v4_balanced.
# Co-located serve + eval so nvidia-smi sees only this GPU and the job
# survives a dropped SSH session.
#
#   sbatch cluster/eval_s2_v4.sbatch
#
# Protocol matches the product router: T=0, seed=42, max_tokens=256,
# --align-prompt fewshot, 3 repeats, gpu_memory_utilization=0.90, bf16.

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

MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0}"
PORT="${PORT:-8002}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

mkdir -p "${ROOT}/logs" "${ROOT}/results/week_9_baselines"

echo "[s2] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[s2] model=${MODEL} port=${PORT} gpu_mem_util=${GPU_MEMORY_UTILIZATION}"
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

echo "[s2] waiting for vLLM pid=${VLLM_PID} on :${PORT}"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[s2] health 200 after ${i} checks"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[s2] vLLM exited before becoming healthy" >&2
    exit 1
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "[s2] TIMEOUT waiting for mamay12 on :${PORT}" >&2
  exit 1
fi

cd "${ROOT}"
python scripts/run_small_router_cluster.py --mode mamay12 \
  --bench benchmarks/mixed_ua_v4_balanced.jsonl \
  --align-prompt fewshot \
  --repeats 3 \
  --out-dir results/week_9_baselines

python scripts/score_repeat_dir.py results/week_9_baselines \
  --out results/week_9_baselines/scores_mean_sd_s2.json

echo "[s2] finished"
