#!/bin/bash
#SBATCH --job-name=mamay27-ref
#SBATCH --account=e-dev-2026d09-051
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --output=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/mamay27-ref-%j.out
#SBATCH --error=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/mamay27-ref-%j.err

# Mamay-27B as the reference model (one big model instead of the router) on JUPITER:
# bf16 on one GH200 card (about 54 GB of weights, 96 GB card), gpu_memory_utilization 0.90, nothing else
# on that card, T=0, seed=42, max_tokens=1024, one run. Same settings as the Mamay-4B / Lapa / Aya runs
# on these sets, so the numbers are comparable with them (JUPITER runs only).
#
# Sets: nodes_mixed_600_test (single steps) and graphs_mixed_100_dev (whole requests, one call).
# The model must be in HF_HOME before the job (compute nodes are offline); on a login node:
#   HF_HOME=/e/project1/e-dev-2026d09-051/onyshchuk1/hf huggingface-cli download INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0
#
#   sbatch cluster/jupiter_mamay27_reference.sbatch
#   BENCHES="graphs_mixed_100_dev" sbatch --export=ALL cluster/jupiter_mamay27_reference.sbatch

set -euo pipefail

ROOT=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_HOME=/e/project1/e-dev-2026d09-051/onyshchuk1/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

BENCHES="${BENCHES:-nodes_mixed_600_test graphs_mixed_100_dev}"
MODEL="INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0"
PORT=8002

echo "[mamay27] host=$(hostname) benches=${BENCHES}"
nvidia-smi -L

CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --enforce-eager > "${ROOT}/logs/serve-mamay27-${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill "${SERVER}" 2>/dev/null || true; wait "${SERVER}" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[mamay27] health 200 :${PORT}"
    break
  fi
  if ! kill -0 "${SERVER}" 2>/dev/null; then
    echo "[mamay27] server exited" >&2
    tail -40 "${ROOT}/logs/serve-mamay27-${SLURM_JOB_ID}.log" >&2
    exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "[mamay27] TIMEOUT" >&2; exit 1; }
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

cd "${ROOT}"
for bench in ${BENCHES}; do
  raw="${ROOT}/logs/${bench}"
  out="${ROOT}/results/${bench}"
  mkdir -p "${raw}" "${out}"
  sha256sum "benchmarks/${bench}.jsonl"
  # The runner and its tokenizer stay off card 0, which holds the model.
  CUDA_VISIBLE_DEVICES=3 python scripts/run_composite_router.py \
    --bench "benchmarks/${bench}.jsonl" \
    --systems mamay27 \
    --repeats 1 \
    --temperature 0 \
    --seed 42 \
    --max-tokens 1024 \
    --out-dir "${raw}"
  newest=$(ls -t "${raw}"/mamay27_composite_"${bench}"_*.jsonl | head -1)
  python scripts/score_composite_results.py "${newest}" --out "${out}/scores_mamay27.json"
  echo "[mamay27] ${bench} -> ${out}/scores_mamay27.json"
done

echo "[mamay27] finished"
