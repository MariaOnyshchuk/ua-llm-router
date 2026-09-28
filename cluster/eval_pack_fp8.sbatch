#!/bin/bash
#SBATCH --job-name=pack-fp8-v4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/Diploma/logs/pack-fp8-v4-%j.out
#SBATCH --error=/home/%u/Diploma/logs/pack-fp8-v4-%j.err

# One-GPU packing screen: Mamay-4B + Lapa-12B + Aya-8B, online FP8.
# gpu_memory_utilization is a fraction of the WHOLE card, so the three
# fractions must sum to under 1. Start the largest model first.
# Ports are off the bf16 defaults so a 3-GPU hybrid job can run beside this.
#
#   sbatch cluster/eval_pack_fp8.sbatch
#
# Protocol: rules v2 + few-shot, mixed_ua_v4, T=0, seed=42, max_tokens=256.
# 1 repeat first. 3 repeats only if that screen beats week-8 bnb (0.830).

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

OUT="${ROOT}/results/week_11_pack_fp8"
mkdir -p "${ROOT}/logs" "${OUT}"

LAPA_PORT=8101
MAMAY_PORT=8103
AYA_PORT=8105
export ROUTER_PORT_LAPA="${LAPA_PORT}"
export ROUTER_PORT_MAMAY4="${MAMAY_PORT}"
export ROUTER_PORT_AYA="${AYA_PORT}"

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

wait_port() {
  local port="$1" name="$2" pid="$3"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[pack] health 200 ${name} :${port}"
      nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[pack] ${name} exited before health on :${port}" >&2
      return 1
    fi
    sleep 10
  done
  echo "[pack] TIMEOUT waiting for ${name} on :${port}" >&2
  return 1
}

smoke() {
  local port="$1" model="$2" name="$3"
  curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Скажи так.\"}],\"max_tokens\":8,\"temperature\":0}" \
    -o "${OUT}/smoke_${name}.json"
  echo "[pack] smoke ${name} ok"
}

serve_one "lapa-llm/lapa-v0.1.2-instruct" "${LAPA_PORT}" "${LAPA_UTIL}" lapa
wait_port "${LAPA_PORT}" lapa "${PIDS[0]}"
serve_one "CohereLabs/aya-expanse-8b" "${AYA_PORT}" "${AYA_UTIL}" aya
wait_port "${AYA_PORT}" aya "${PIDS[1]}"
serve_one "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" "${MAMAY_PORT}" "${MAMAY_UTIL}" mamay4
wait_port "${MAMAY_PORT}" mamay4 "${PIDS[2]}"

{
  echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "utils lapa=${LAPA_UTIL} aya=${AYA_UTIL} mamay4=${MAMAY_UTIL}"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
} | tee "${OUT}/nvidia_after_load.txt"

smoke "${LAPA_PORT}" "lapa-llm/lapa-v0.1.2-instruct" lapa
smoke "${AYA_PORT}" "CohereLabs/aya-expanse-8b" aya
smoke "${MAMAY_PORT}" "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" mamay4

cd "${ROOT}"
python scripts/run_small_router_cluster.py --mode router_matrix \
  --bench benchmarks/mixed_ua_v4_balanced.jsonl \
  --align-prompt fewshot \
  --repeats 1 \
  --out-dir results/week_11_pack_fp8

SCREEN=$(find "${OUT}" -maxdepth 1 -name 'router_matrix_v2_fewshot_*.jsonl' -print -quit)
python scripts/score_results.py "${SCREEN}" --out "${OUT}/scores_screen.json"

OVERALL=$(python -c "
import json
p = json.load(open('${OUT}/scores_screen.json'))
sys = p['systems']['router_matrix_v2_fewshot']
print(sys['overall_mean'])
")
echo "[pack] screen overall=${OVERALL} file=${SCREEN}"

# Week-8 bitsandbytes was 0.830. 3x only if this screen beats that.
if python -c "import sys; sys.exit(0 if float('${OVERALL}') > 0.830 else 1)"; then
  echo "[pack] screen beats 0.830; running 3 repeats"
else
  echo "[pack] screen ${OVERALL} does not beat 0.830; stop after 1 repeat"
  exit 0
fi

python scripts/run_small_router_cluster.py --mode router_matrix \
  --bench benchmarks/mixed_ua_v4_balanced.jsonl \
  --align-prompt fewshot \
  --repeats 3 \
  --out-dir results/week_11_pack_fp8

python scripts/score_repeat_dir.py results/week_11_pack_fp8 \
  --out results/week_11_pack_fp8/scores_mean_sd_pack.json

echo "[pack] finished"
