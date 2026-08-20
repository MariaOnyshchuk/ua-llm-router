#!/bin/bash
#SBATCH --job-name=serve-qwen7b
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-qwen7b-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-qwen7b-%j.err

# Larger cross-lineage code specialist.
# Submit: sbatch cluster/serve_qwen7b.sbatch
# Port 8004 replaces Qwen-Coder-3B while this job is live.

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT="${PORT:-8004}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[serve-qwen7b] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-qwen7b] model=${MODEL} port=${PORT} gpu_mem_util=${GPU_MEMORY_UTILIZATION}"
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
