#!/bin/bash
#SBATCH --job-name=graphs-100-dev
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/Diploma/logs/graphs-100-dev-%j.out
#SBATCH --error=/home/%u/Diploma/logs/graphs-100-dev-%j.err

# Composite graphs_mixed_100_dev (benchmarks/graphs_mixed_100_dev.jsonl): one call, the gated
# planner with Mamay-4B, Lapa or Aya writing the plan, and the gold graph. All 100 items are a
# development set; claims will come from a separate test set.
# Specialists in bf16, one per card. T=0, seed=42, max_tokens=1024, one run.
#
#   sbatch cluster/eval_graphs_mixed_100_dev.sbatch
#
# Optional env:
#   SYSTEMS=router_direct,hybrid_gated,hybrid_gated_lapa,hybrid_gated_aya,oracle
#   SPLIT=all           (or a split name stored in the items)
#   MAMAY_UTIL=0.90     lower it (e.g. 0.55) when Slurm gives a card that another
#                       user's process already holds; Mamay-4B is served first.
#
# Raw rows (prompt/content) stay in logs/graphs_mixed_100_dev; only scores go to results/.

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

SYSTEMS="${SYSTEMS:-router_direct,hybrid_gated,hybrid_gated_lapa,hybrid_gated_aya,oracle}"
SPLIT="${SPLIT:-all}"
SPLIT_ARG=(--split "")
if [[ "${SPLIT}" != "all" ]]; then
  SPLIT_ARG=(--split "${SPLIT}")
fi
RAW_DIR="${ROOT}/logs/graphs_mixed_100_dev"
OUT_DIR="${ROOT}/results/graphs_mixed_100_dev"
mkdir -p "${ROOT}/logs" "${RAW_DIR}" "${OUT_DIR}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[graphs_100_dev] need 3 GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

echo "[graphs_100_dev] host=$(hostname) gpus=${GPUS[*]} systems=${SYSTEMS} split=${SPLIT}"
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
  local gpu="$1" model="$2" port="$3" name="$4" util="${5:-0.90}"
  echo "[graphs_100_dev] ${name} gpu=${gpu} port=${port} util=${util}"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization "${util}" \
    --trust-remote-code \
    --enforce-eager &
  PIDS+=("$!")
}

serve_one "${GPUS[0]}" "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003 mamay4 "${MAMAY_UTIL:-0.90}"
serve_one "${GPUS[1]}" "lapa-llm/lapa-v0.1.2-instruct" 8001 lapa
serve_one "${GPUS[2]}" "CohereLabs/aya-expanse-8b" 8005 aya

wait_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[graphs_100_dev] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[graphs_100_dev] TIMEOUT waiting for ${name} on :${port}" >&2
  return 1
}

wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya

cd "${ROOT}"
sha256sum benchmarks/graphs_mixed_100_dev.jsonl

python scripts/run_composite_router.py \
  --bench benchmarks/graphs_mixed_100_dev.jsonl \
  --systems "${SYSTEMS}" \
  "${SPLIT_ARG[@]}" \
  --repeats 1 \
  --temperature 0 \
  --seed 42 \
  --max-tokens 1024 \
  --out-dir "${RAW_DIR}"

python scripts/score_composite_results.py "${RAW_DIR}"/*_composite_graphs_mixed_100_dev_*.jsonl \
  --out "${OUT_DIR}/scores_${SPLIT}.json"

echo "[graphs_100_dev] finished → ${OUT_DIR}/scores_${SPLIT}.json"
