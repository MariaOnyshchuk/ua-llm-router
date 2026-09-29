#!/bin/bash
#SBATCH --job-name=agentcoma-solo
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/Diploma/logs/agentcoma-solo-%x-%j.out
#SBATCH --error=/home/%u/Diploma/logs/agentcoma-solo-%x-%j.err

# One model alone on AgentCoMa-UK 200: (a) the composite prompt answered directly,
# (b) the two step questions asked separately (for the compositional gap).
# One model live on the GPU, bf16 (mamay27 in FP8), gpu_memory_utilization 0.90.
# T=0, seed=42, max_tokens=256, one pass. Default SPLIT=dev; run SPLIT=test once,
# only after the config is frozen.
#
#   sbatch --job-name=ac-mamay12 --export=ALL,ALIAS=mamay12 cluster/eval_agentcoma_solo.sbatch
#   sbatch --job-name=ac-mamay27 --export=ALL,ALIAS=mamay27 cluster/eval_agentcoma_solo.sbatch
#   (also ALIAS=mamay4|lapa|aya;  SPLIT=test for the test run)
#
# benchmarks/agentcoma_uk_200_{composite,merged}.jsonl are not in git: scp them first.
# Generation JSONL stays on the lab; pull back only scores and *_detail.jsonl.

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

ALIAS="${ALIAS:-mamay12}"
SPLIT="${SPLIT:-dev}"
QUANT=""
case "${ALIAS}" in
  mamay4)  MODEL="INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0"; PORT=8003 ;;
  lapa)    MODEL="lapa-llm/lapa-v0.1.2-instruct"; PORT=8001 ;;
  aya)     MODEL="CohereLabs/aya-expanse-8b"; PORT=8005 ;;
  mamay12) MODEL="INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0"; PORT=8002 ;;
  mamay27) MODEL="INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0"; PORT=8002; QUANT="fp8" ;;
  *) echo "unknown ALIAS=${ALIAS}" >&2; exit 1 ;;
esac
QUANT_ARGS=()
[ -n "${QUANT}" ] && QUANT_ARGS=(--quantization "${QUANT}")

COMPOSITE="${ROOT}/benchmarks/agentcoma_uk_200_composite.jsonl"
MERGED="${ROOT}/benchmarks/agentcoma_uk_200_merged.jsonl"
COMP_DIR="${ROOT}/results/week_13_agentcoma_200_${SPLIT}"
STEP_DIR="${ROOT}/results/week_13_agentcoma_steps/${SPLIT}"
mkdir -p "${ROOT}/logs" "${COMP_DIR}" "${STEP_DIR}"
for f in "${COMPOSITE}" "${MERGED}"; do
  [[ -f "${f}" ]] || { echo "[ac-solo] missing ${f}; scp it first" >&2; exit 1; }
done

# The node is shared with users outside our Slurm allocation, so an assigned card can already
# hold memory (seen on 29 Sep 2026: card 0 had ~22 GB from another user). Refuse to start then;
# do not override CUDA_VISIBLE_DEVICES by hand, the other cards belong to running jobs.
GPU_FIRST="${CUDA_VISIBLE_DEVICES:-0}"; GPU_FIRST="${GPU_FIRST%,*}"
GPU_USED_MB="$(nvidia-smi -i "${GPU_FIRST}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "${GPU_USED_MB:-0}" -gt 2000 ]; then
  echo "[ac-solo] GPU ${GPU_FIRST} already holds ${GPU_USED_MB} MiB; free it or submit with CUDA_VISIBLE_DEVICES=<free card>" >&2
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >&2
  exit 1
fi
echo "[ac-solo] host=$(hostname) alias=${ALIAS} split=${SPLIT} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

vllm serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --trust-remote-code \
  --enforce-eager "${QUANT_ARGS[@]}" &
VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && { echo "[ac-solo] health 200"; break; }
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "[ac-solo] vLLM exited early" >&2; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "[ac-solo] TIMEOUT" >&2; exit 1; }

cd "${ROOT}"
python scripts/run_composite_router.py --bench "${COMPOSITE}" --systems "${ALIAS}" \
  --split "${SPLIT}" --repeats 1 --temperature 0 --seed 42 --max-tokens 256 \
  --out-dir "${COMP_DIR}"
python scripts/run_agentcoma_steps.py run --alias "${ALIAS}" --split "${SPLIT}" \
  --out-dir "results/week_13_agentcoma_steps/${SPLIT}"

python scripts/score_composite_results.py "${COMP_DIR}/${ALIAS}"_composite_agentcoma200_*.jsonl \
  --out "${COMP_DIR}/scores_${ALIAS}.json"
python scripts/run_agentcoma_steps.py score "${STEP_DIR}/${ALIAS}"_steps_agentcoma200_*.jsonl \
  --out "${STEP_DIR}/scores_steps_${ALIAS}.json"
echo "[ac-solo] finished ${ALIAS} ${SPLIT}"
