#!/bin/bash
#SBATCH --job-name=serve-aya
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-aya-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-aya-%j.err

# Cross-lineage third specialist (Cohere Command / Aya Expanse 8B).
# Submit: sbatch cluster/serve_aya.sbatch
# Tunnel: ssh -L 8005:localhost:8005 ucu-lab-2240

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

MODEL="${MODEL:-CohereLabs/aya-expanse-8b}"
PORT="${PORT:-8005}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[serve-aya] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-aya] model=${MODEL} port=${PORT} gpu_mem_util=${GPU_MEMORY_UTILIZATION}"
nvidia-smi -L

export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --enforce-eager
