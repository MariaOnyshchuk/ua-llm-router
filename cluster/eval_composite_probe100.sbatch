#!/bin/bash
#SBATCH --job-name=comp-probe100
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/Diploma/logs/comp-probe100-%j.out
#SBATCH --error=/home/%u/Diploma/logs/comp-probe100-%j.err

# Composite probe-100 (benchmarks/archive/mixed_ua_composite_probe100.jsonl): planning and
# step execution on 100 new composite requests. Probe only, not for claims.
# Specialists in bf16, one per card. T=0, seed=42, max_tokens=1024, one run.
#
#   sbatch cluster/eval_composite_probe100.sbatch
#
# Optional env:
#   SYSTEMS=router_direct,template,hybrid_gated,oracle
#   MAMAY_UTIL=0.90     lower it (e.g. 0.55) when Slurm gives a card that another
#                       user's process already holds; Mamay-4B is served first.
#
# Raw rows (prompt/content) stay in logs/composite_probe100; only scores go to results/.

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

SYSTEMS="${SYSTEMS:-router_direct,template,hybrid_gated,oracle}"
RAW_DIR="${ROOT}/logs/composite_probe100"
OUT_DIR="${ROOT}/results/composite_probe100"
mkdir -p "${ROOT}/logs" "${RAW_DIR}" "${OUT_DIR}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[probe100] need 3 GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

echo "[probe100] host=$(hostname) gpus=${GPUS[*]} systems=${SYSTEMS}"
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
  echo "[probe100] ${name} gpu=${gpu} port=${port} util=${util}"
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
      echo "[probe100] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[probe100] TIMEOUT waiting for ${name} on :${port}" >&2
  return 1
}

wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya

cd "${ROOT}"
python scripts/benchmarks/build_composite_probe100.py --check

python scripts/run_composite_router.py \
  --bench benchmarks/archive/mixed_ua_composite_probe100.jsonl \
  --systems "${SYSTEMS}" \
  --repeats 1 \
  --temperature 0 \
  --seed 42 \
  --max-tokens 1024 \
  --out-dir "${RAW_DIR}"

python scripts/score_composite_results.py "${RAW_DIR}"/*_composite_probe100_*.jsonl \
  --out "${OUT_DIR}/scores.json"

echo "[probe100] finished → ${OUT_DIR}/scores.json"
