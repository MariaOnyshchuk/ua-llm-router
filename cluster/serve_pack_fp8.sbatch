#!/bin/bash
#SBATCH --job-name=serve-pack-fp8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/Diploma/logs/serve-pack-fp8-%j.out
#SBATCH --error=/home/%u/Diploma/logs/serve-pack-fp8-%j.err

# Serve the three specialists on ONE card for the demo page and for manual testing:
# Mamay-4B + Lapa-12B + Aya-8B in online FP8, on the usual ports 8001 / 8003 / 8005.
# gpu_memory_utilization is a fraction of the WHOLE card; the three fractions sum to 0.88,
# so the card must be empty. Same settings as the measured pack run (week 11): FP8,
# max-model-len 4096. Not the bf16 configuration of the claim runs: do not use it for those.
#
#   sbatch cluster/serve_pack_fp8.sbatch
#   ssh -L 8001:localhost:8001 -L 8003:localhost:8003 -L 8005:localhost:8005 ucu-lab-2240

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

mkdir -p "${ROOT}/logs"

LAPA_PORT=8001
MAMAY_PORT=8003
AYA_PORT=8005

# Fractions of the full 48 GiB card. Sum 0.88.
LAPA_UTIL=0.42
AYA_UTIL=0.28
MAMAY_UTIL=0.18

echo "[pack] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

serve_one() {
  local model="$1" port="$2" util="$3" name="$4"
  echo "[pack] start ${name} port=${port} util=${util} quant=fp8"
  vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --quantization fp8 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization "${util}" \
    --trust-remote-code \
    --enforce-eager &
  PIDS+=("$!")
}

GPU_FIRST="${CUDA_VISIBLE_DEVICES:-0}"; GPU_FIRST="${GPU_FIRST%,*}"
GPU_USED_MB="$(nvidia-smi -i "${GPU_FIRST}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${GPU_USED_MB:-0}" -gt 2000 ]; then
  echo "[pack] GPU ${GPU_FIRST} already holds ${GPU_USED_MB} MiB; the pack needs an empty card" >&2
  exit 1
fi

# Largest model first; each one must be up before the next asks for its share of the card.
start_and_wait() {
  local model="$1" port="$2" util="$3" name="$4"
  serve_one "${model}" "${port}" "${util}" "${name}"
  local pid="${PIDS[-1]}"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[pack] health 200 ${name} :${port}"
      return 0
    fi
    kill -0 "${pid}" 2>/dev/null || { echo "[pack] ${name} exited before becoming healthy" >&2; exit 1; }
    sleep 10
  done
  echo "[pack] TIMEOUT ${name}" >&2
  exit 1
}

start_and_wait "lapa-llm/lapa-v0.1.2-instruct" "${LAPA_PORT}" "${LAPA_UTIL}" lapa
start_and_wait "CohereLabs/aya-expanse-8b" "${AYA_PORT}" "${AYA_UTIL}" aya
start_and_wait "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" "${MAMAY_PORT}" "${MAMAY_UTIL}" mamay4
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "[pack] all three up; serving until the job ends"
wait
