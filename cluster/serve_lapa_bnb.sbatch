#!/bin/bash
#SBATCH --job-name=serve-lapa-bnb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-lapa-bnb-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-lapa-bnb-%j.err

# 4-bit bitsandbytes serve of Lapa-12B (same HF weights as bf16 :8001).
# Submit: sbatch cluster/serve_lapa_bnb.sbatch

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

MODEL="${MODEL:-lapa-llm/lapa-v0.1.2-instruct}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[serve-lapa-bnb] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-lapa-bnb] model=${MODEL} port=${PORT} quant=bitsandbytes util=${GPU_MEMORY_UTILIZATION}"
nvidia-smi -L

export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --enforce-eager
