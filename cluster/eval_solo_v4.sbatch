#!/bin/bash
#SBATCH --job-name=solo-v4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/Diploma/logs/solo-v4-%j.out
#SBATCH --error=/home/%u/Diploma/logs/solo-v4-%j.err

# One model alone on mixed_ua_v4_balanced, optionally with few-shot examples in
# selected buckets. Serve and eval run in one job, so nvidia-smi sees only this
# GPU and the job survives a dropped SSH session.
#
# Required:
#   MODE=mamay4|lapa|aya|mamay12|mamay27
# Optional (defaults in brackets):
#   BUCKET_FEWSHOT  comma list of chat,code,translate,instruct,knowledge or all  [none]
#   ALIGN_PROMPT    baseline|fewshot                                              [fewshot]
#   REPEATS         number of repeats                                             [1]
#   QUANT           vLLM quantization, e.g. fp8                                   [none; fp8 for mamay27]
#   GPU_MEMORY_UTILIZATION                                                        [0.90]
#   OUT_DIR         results folder                       [results/week_13_solo_v4/<MODE>_<tag>]
#
# Examples:
#   MODE=mamay27 sbatch cluster/eval_solo_v4.sbatch                       # S2b: 27B, FP8 (bf16 needs ~54 GB, does not fit one 49 GB card)
#   MODE=lapa BUCKET_FEWSHOT=knowledge,instruct sbatch cluster/eval_solo_v4.sbatch
#   MODE=aya BUCKET_FEWSHOT=all sbatch cluster/eval_solo_v4.sbatch
#
# Decoding: T=0, seed=42, max_tokens=256 (defaults of run_small_router_cluster.py).
# A 27B run with QUANT=fp8 is a different setup from the bf16 models: report it that way.

set -euo pipefail

: "${MODE:?set MODE=mamay4|lapa|aya|mamay12|mamay27}"
BUCKET_FEWSHOT="${BUCKET_FEWSHOT:-}"
ALIGN_PROMPT="${ALIGN_PROMPT:-fewshot}"
REPEATS="${REPEATS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
QUANT="${QUANT:-}"

case "${MODE}" in
  mamay4)  MODEL="INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0";  PORT=8003 ;;
  lapa)    MODEL="lapa-llm/lapa-v0.1.2-instruct";                PORT=8001 ;;
  aya)     MODEL="CohereLabs/aya-expanse-8b";                    PORT=8005 ;;
  mamay12) MODEL="INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0"; PORT=8002 ;;
  mamay27) MODEL="INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0"; PORT=8002; QUANT="${QUANT:-fp8}" ;;
  *) echo "[solo] unknown MODE=${MODE}" >&2; exit 2 ;;
esac

TAG="${BUCKET_FEWSHOT:-plain}"
TAG="${TAG//,/-}"
[ -n "${QUANT}" ] && TAG="${TAG}_${QUANT}"
OUT_DIR="${OUT_DIR:-results/week_13_solo_v4/${MODE}_${TAG}}"

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

mkdir -p "${ROOT}/logs" "${ROOT}/${OUT_DIR}"

echo "[solo] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[solo] mode=${MODE} model=${MODEL} port=${PORT} quant=${QUANT:-none} util=${GPU_MEMORY_UTILIZATION}"
echo "[solo] bucket_fewshot=${BUCKET_FEWSHOT:-none} align=${ALIGN_PROMPT} repeats=${REPEATS} out=${OUT_DIR}"
nvidia-smi -L

QUANT_ARGS=()
[ -n "${QUANT}" ] && QUANT_ARGS=(--quantization "${QUANT}")

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

echo "[solo] waiting for vLLM pid=${VLLM_PID} on :${PORT}"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[solo] health 200 after ${i} checks"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[solo] vLLM exited before becoming healthy" >&2
    exit 1
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "[solo] TIMEOUT waiting for ${MODE} on :${PORT}" >&2
  exit 1
fi

cd "${ROOT}"
python scripts/run_small_router_cluster.py --mode "${MODE}" \
  --bench benchmarks/mixed_ua_v4_balanced.jsonl \
  --align-prompt "${ALIGN_PROMPT}" \
  --bucket-fewshot "${BUCKET_FEWSHOT}" \
  --repeats "${REPEATS}" \
  --out-dir "${OUT_DIR}"

# With REPEATS=1 file names carry no _rep suffix, so match every jsonl in this run folder.
if [ "${REPEATS}" -gt 1 ]; then PATTERN='*_rep*.jsonl'; else PATTERN='*.jsonl'; fi
python scripts/score_repeat_dir.py "${OUT_DIR}" \
  --pattern "${PATTERN}" \
  --out "${OUT_DIR}/scores_mean_sd_${MODE}_${TAG}.json"

echo "[solo] finished"
