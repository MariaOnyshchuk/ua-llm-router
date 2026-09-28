#!/bin/bash
#SBATCH --job-name=hybrid-v3
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/Diploma/logs/hybrid-v3-%j.out
#SBATCH --error=/home/%u/Diploma/logs/hybrid-v3-%j.err

# Hybrid planner v3: same 36×3 composite suite, with missing {{step.content}}
# spliced in before execution. Needs Mamay-4B, Lapa, and Aya at once.
#
#   sbatch cluster/eval_hybrid_v3.sbatch

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

mkdir -p "${ROOT}/logs" "${ROOT}/results/week_10_composite_hybrid_v3"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[hybrid-v3] need 3 GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

echo "[hybrid-v3] host=$(hostname) gpus=${GPUS[*]}"
nvidia-smi -L

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
  local gpu="$1" model="$2" port="$3" name="$4"
  echo "[hybrid-v3] ${name} gpu=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enforce-eager &
  PIDS+=("$!")
}

serve_one "${GPUS[0]}" "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003 mamay4
serve_one "${GPUS[1]}" "lapa-llm/lapa-v0.1.2-instruct" 8001 lapa
serve_one "${GPUS[2]}" "CohereLabs/aya-expanse-8b" 8005 aya

wait_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[hybrid-v3] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[hybrid-v3] TIMEOUT waiting for ${name} on :${port}" >&2
  return 1
}

wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya

cd "${ROOT}"
python scripts/run_composite_router.py \
  --systems hybrid \
  --repeats 3 \
  --out-dir results/week_10_composite_hybrid_v3

python scripts/score_composite_results.py \
  results/week_10_composite_hybrid_v3/hybrid_*.jsonl \
  --out results/week_10_composite_hybrid_v3/scores.json

echo "[hybrid-v3] finished"
