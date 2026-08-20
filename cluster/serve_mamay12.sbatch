#!/bin/bash
#SBATCH --job-name=serve-mamay12
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-mamay12-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-mamay12-%j.err

# S2 baseline: MamayLM v2.0 12B (same recipe as Lapa serve).
# Submit: sbatch cluster/serve_mamay12.sbatch
# Tunnel: ssh -L 8002:localhost:8002 ucu-lab-2240

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0}"
PORT="${PORT:-8002}"
HOST="${HOST:-0.0.0.0}"
# Fair single-model runs: identical util across systems (override lower to pack).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[serve-mamay12] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-mamay12] model=${MODEL} port=${PORT} gpu_mem_util=${GPU_MEMORY_UTILIZATION}"
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
