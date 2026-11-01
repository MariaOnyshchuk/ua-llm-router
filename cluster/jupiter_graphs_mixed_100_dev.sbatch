#!/bin/bash
#SBATCH --job-name=graphs-100-dev
#SBATCH --account=e-dev-2026d09-051
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --output=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/graphs-100-dev-%j.out
#SBATCH --error=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/graphs-100-dev-%j.err

# Composite graphs_mixed_100_dev on JUPITER (JSC, 4x GH200 96 GB per node, aarch64).
# Same run as cluster/eval_graphs_mixed_100_dev.sbatch on the lab: Mamay-4B, Lapa and Aya
# in bf16, one per card, T=0, seed=42, max_tokens=1024, one run, all 100 items (dev).
# Compute nodes have no internet: models are downloaded beforehand into HF_HOME on a
# login node, and the job runs with HF_HUB_OFFLINE=1.
# Numbers from this machine are compared only with runs from this machine.
#
#   sbatch cluster/jupiter_graphs_mixed_100_dev.sbatch
#
# Optional env: SYSTEMS, SPLIT (see the lab script). Learned routing (router/artifacts/{knn,mlp},
# scripts/train_routerbench_router.py) needs intfloat/multilingual-e5-large in HF_HOME:
#   SYSTEMS=router_knn,router_mlp,oracle+knn,oracle+mlp,hybrid_gated_aya+knn,hybrid_gated_aya+mlp,mamay4,lapa,aya
# The runner (and its e5 encoder) uses the fourth card, so it never shares one with a model.

set -euo pipefail

ROOT=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_HOME=/e/project1/e-dev-2026d09-051/onyshchuk1/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

SYSTEMS="${SYSTEMS:-router_direct,hybrid_gated,hybrid_gated_lapa,hybrid_gated_aya,oracle}"
SPLIT="${SPLIT:-all}"
SPLIT_ARG=(--split "")
if [[ "${SPLIT}" != "all" ]]; then
  SPLIT_ARG=(--split "${SPLIT}")
fi
RAW_DIR="${ROOT}/logs/graphs_mixed_100_dev"
OUT_DIR="${ROOT}/results/graphs_mixed_100_dev"
mkdir -p "${RAW_DIR}" "${OUT_DIR}"

echo "[graphs_100_dev] host=$(hostname) systems=${SYSTEMS} split=${SPLIT}"
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
  echo "[graphs_100_dev] ${name} gpu=${gpu} port=${port}"
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
      echo "[graphs_100_dev] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[graphs_100_dev] TIMEOUT waiting for ${name} on :${port}" >&2
  tail -30 "${ROOT}/logs/serve-${name}-${SLURM_JOB_ID}.log" >&2
  return 1
}

wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya

cd "${ROOT}"
sha256sum benchmarks/graphs_mixed_100_dev.jsonl

CUDA_VISIBLE_DEVICES=3 python scripts/run_composite_router.py \
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

echo "[graphs_100_dev] finished -> ${OUT_DIR}/scores_${SPLIT}.json"
