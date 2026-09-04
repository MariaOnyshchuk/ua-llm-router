#!/bin/bash
#SBATCH --job-name=serve-m4-bnb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-mamay4-bnb-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-mamay4-bnb-%j.err

# 4-bit bitsandbytes serve of Mamay-4B (same HF weights as bf16 :8003).
# Submit: sbatch cluster/serve_mamay4_bnb.sbatch

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0}"
PORT="${PORT:-8003}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[serve-mamay4-bnb] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-mamay4-bnb] model=${MODEL} port=${PORT} quant=bitsandbytes util=${GPU_MEMORY_UTILIZATION}"
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
  --limit-mm-per-prompt '{"image":0}' \
  --trust-remote-code \
  --enforce-eager
