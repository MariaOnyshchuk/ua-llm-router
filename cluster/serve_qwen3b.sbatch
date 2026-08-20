#!/bin/bash
#SBATCH --job-name=serve-qwen3b
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-qwen3b-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-qwen3b-%j.err

# Small code specialist for the router.
# Submit: sbatch cluster/serve_qwen3b.sbatch
# Tunnel: ssh -L 8004:localhost:8004 ucu-lab-2240

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
PORT="${PORT:-8004}"
HOST="${HOST:-0.0.0.0}"
# Fair single-model runs: identical util across systems (override lower to pack).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[serve-qwen3b] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-qwen3b] model=${MODEL} port=${PORT} gpu_mem_util=${GPU_MEMORY_UTILIZATION}"
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
