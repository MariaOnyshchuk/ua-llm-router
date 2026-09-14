#!/bin/bash
#SBATCH --job-name=serve-lapa
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-lapa-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-lapa-%j.err

# Serve Lapa (OpenAI-compatible) on port 8001 via vLLM.
# Submit: sbatch cluster/serve_lapa.sbatch
# Then from laptop: ssh -L 8001:localhost:8001 ucu-lab-2240

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

MODEL="${MODEL:-lapa-llm/lapa-v0.1.2-instruct}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
# Fair single-model runs: identical util across systems (override lower to pack).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "[serve-lapa] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-lapa] model=${MODEL} port=${PORT} gpu_mem_util=${GPU_MEMORY_UTILIZATION}"
nvidia-smi -L

# FlashInfer sampler JIT fails on this Ada box (sm_89) with vLLM 0.25.x —
# force PyTorch sampler for a reliable first bring-up.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Single GPU, bf16 — 12B fits comfortably on 48GB
exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --enforce-eager
