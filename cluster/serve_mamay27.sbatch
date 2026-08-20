#!/bin/bash
#SBATCH --job-name=serve-mamay27
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-mamay27-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-mamay27-%j.err

# S2 baseline: MamayLM v2.0 27B in 4-bit (bitsandbytes) on one 48GB GPU.
# TP=2 bf16 crashed in vLLM multiproc on this box; quantized single-GPU is the reliable path.
# Submit: sbatch cluster/serve_mamay27.sbatch
# Tunnel: ssh -L 8002:localhost:8002 ucu-lab-2240

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0}"
PORT="${PORT:-8002}"
HOST="${HOST:-0.0.0.0}"

echo "[serve-mamay27] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[serve-mamay27] model=${MODEL} port=${PORT} quant=bitsandbytes"
nvidia-smi -L

export VLLM_USE_FLASHINFER_SAMPLER=0

pip install -q bitsandbytes

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":0}' \
  --trust-remote-code \
  --enforce-eager
