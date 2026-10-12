#!/bin/bash
#SBATCH --job-name=v7-solo
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/Diploma/logs/v7-solo-%x-%j.out
#SBATCH --error=/home/%u/Diploma/logs/v7-solo-%x-%j.err

# One model alone on mixed_ua_v7 (1 pass, T=0, seed=42, max_tokens=1024), co-located serve + eval.
# Default SPLIT=dev. Run SPLIT=test ONCE, only after the router is frozen (AGENTS.md).
#
#   sbatch --job-name=v7-mamay12 --export=ALL,ALIAS=mamay12 cluster/eval_v7_solo.sbatch              # dev
#   sbatch --job-name=v7-mamay12 --export=ALL,ALIAS=mamay12,SPLIT=test cluster/eval_v7_solo.sbatch
#   (ALIAS = mamay4 | lapa | aya | mamay12 | mamay27 (FP8) | qwen7)
#
# Needs benchmarks/mixed_ua_v7.jsonl and the current scripts/run_small_router_cluster.py on the
# lab (scp + checksum). Writes results/v7_<split>_solos/. Not yet run: try SPLIT=dev first.

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
SPLIT="${SPLIT:-dev}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
BENCH="${BENCH:-benchmarks/mixed_ua_v7.jsonl}"
OUT_DIR="${ROOT}/results/v7_${SPLIT}_solos"
mkdir -p "${ROOT}/logs" "${OUT_DIR}"
if [[ ! -f "${ROOT}/${BENCH}" ]]; then
  echo "[v7-solo] missing ${BENCH}; scp it from the laptop first" >&2
  exit 1
fi

ALIAS="${ALIAS:-mamay12}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_USE_FLASHINFER_SAMPLER=0

case "${ALIAS}" in
  mamay4)
    MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0}"
    PORT="${PORT:-8003}"
    ;;
  lapa)
    MODEL="${MODEL:-lapa-llm/lapa-v0.1.2-instruct}"
    PORT="${PORT:-8001}"
    ;;
  aya)
    MODEL="${MODEL:-CohereLabs/aya-expanse-8b}"
    PORT="${PORT:-8005}"
    ;;
  mamay12)
    MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0}"
    PORT="${PORT:-8002}"
    ;;
  mamay27)
    # bf16 needs about 54 GB and does not fit one 49 GB card; serve in FP8 and report it as such.
    MODEL="${MODEL:-INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0}"
    PORT="${PORT:-8002}"
    QUANT="${QUANT:-fp8}"
    ;;
  qwen7)
    MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    PORT="${PORT:-8004}"
    ;;
  *)
    echo "unknown ALIAS=${ALIAS} (expected mamay4|lapa|aya|mamay12|mamay27|qwen7)" >&2
    exit 1
    ;;
esac

QUANT="${QUANT:-}"
QUANT_ARGS=()
[ -n "${QUANT}" ] && QUANT_ARGS=(--quantization "${QUANT}")

# The node is shared with users outside our Slurm allocation, so an assigned card can already
# hold memory (seen on 29 Sep 2026: card 0 had ~22 GB from another user). Refuse to start then;
# do not override CUDA_VISIBLE_DEVICES by hand, the other cards belong to running jobs.
GPU_FIRST="${CUDA_VISIBLE_DEVICES:-0}"; GPU_FIRST="${GPU_FIRST%,*}"
GPU_USED_MB="$(nvidia-smi -i "${GPU_FIRST}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${GPU_USED_MB:-0}" -gt 2000 ]; then
  echo "[v7-solo] GPU ${GPU_FIRST} already holds ${GPU_USED_MB} MiB; free it or submit with CUDA_VISIBLE_DEVICES=<free card>" >&2
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >&2
  exit 1
fi
echo "[v7-solo] host=$(hostname) alias=${ALIAS} split=${SPLIT} port=${PORT} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

vllm serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --enforce-eager \
  "${QUANT_ARGS[@]}" &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[v7-solo] waiting for vLLM pid=${VLLM_PID} on :${PORT}"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[v7-solo] health 200 after ${i} checks"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[v7-solo] vLLM exited before becoming healthy" >&2
    exit 1
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "[v7-solo] TIMEOUT waiting for ${ALIAS} on :${PORT}" >&2
  exit 1
fi

cd "${ROOT}"
python scripts/run_small_router_cluster.py --mode "${ALIAS}" --bench "${BENCH}" --split "${SPLIT}" \
  --repeats 1 --max-tokens "${MAX_TOKENS}" --out-dir "${OUT_DIR}"
python scripts/score_results.py "${OUT_DIR}/${ALIAS}"_mixed_ua_v7_"${SPLIT}"_*.jsonl \
  --out "${OUT_DIR}/scores_${ALIAS}.json"
echo "[v7-solo] finished ${ALIAS} ${SPLIT}"
