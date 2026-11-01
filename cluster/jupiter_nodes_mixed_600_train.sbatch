#!/bin/bash
#SBATCH --job-name=nodes_mixed_600_train
#SBATCH --account=e-dev-2026d09-051
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --output=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/nodes-train-%j.out
#SBATCH --error=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/nodes-train-%j.err

# Node training set (benchmarks/nodes_mixed_600_train.jsonl, 600 single-skill items) on JUPITER:
# Mamay-4B, Lapa and Aya in bf16, one per card, T=0, seed=42, max_tokens=1024, one run.
# The scores are training labels for the learned routers (kNN, MLP), not results to cite.
# Each model has its own runner process, so the three run at the same time.
#
#   sbatch cluster/jupiter_nodes_mixed_600_train.sbatch

set -euo pipefail

ROOT=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_HOME=/e/project1/e-dev-2026d09-051/onyshchuk1/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

RAW_DIR="${ROOT}/logs/nodes_mixed_600_train"
OUT_DIR="${ROOT}/results/nodes_mixed_600_train"
BENCH=benchmarks/nodes_mixed_600_train.jsonl
mkdir -p "${RAW_DIR}" "${OUT_DIR}"

echo "[nodes] host=$(hostname)"
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
  echo "[nodes] ${name} gpu=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enforce-eager > "${ROOT}/logs/serve-${name}-${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=("$!")
}

serve_one 0 "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003 mamay4
serve_one 1 "lapa-llm/lapa-v0.1.2-instruct" 8001 lapa
serve_one 2 "CohereLabs/aya-expanse-8b" 8005 aya

wait_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[nodes] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[nodes] TIMEOUT waiting for ${name} on :${port}" >&2
  tail -30 "${ROOT}/logs/serve-${name}-${SLURM_JOB_ID}.log" >&2
  return 1
}

wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya

cd "${ROOT}"
sha256sum "${BENCH}"

RUNNERS=()
for system in mamay4 lapa aya; do
  python scripts/run_composite_router.py \
    --bench "${BENCH}" \
    --systems "${system}" \
    --repeats 1 \
    --temperature 0 \
    --seed 42 \
    --max-tokens 1024 \
    --out-dir "${RAW_DIR}" > "${ROOT}/logs/nodes-run-${system}-${SLURM_JOB_ID}.log" 2>&1 &
  RUNNERS+=("$!")
done
status=0
for pid in "${RUNNERS[@]}"; do
  wait "${pid}" || status=1
done
for log in "${ROOT}"/logs/nodes-run-*-"${SLURM_JOB_ID}".log; do tail -n 2 "${log}"; done
[[ "${status}" == 0 ]] || { echo "[nodes] a runner failed" >&2; exit 1; }

python scripts/score_composite_results.py "${RAW_DIR}"/*_composite_nodes_mixed_600_train_*.jsonl \
  --out "${OUT_DIR}/scores_all.json"

echo "[nodes] finished -> ${OUT_DIR}/scores_all.json"
