#!/bin/bash
#SBATCH --job-name=v7-router
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/Diploma/logs/v7-router-%j.out
#SBATCH --error=/home/%u/Diploma/logs/v7-router-%j.err

# Rules v2 router (alignment few-shot) on mixed_ua_v7, 1 pass, T=0, seed=42, max_tokens=1024.
# Serves the three specialists on three GPUs (bf16, gpu_memory_utilization 0.90).
# Default SPLIT=dev. Run SPLIT=test ONCE, only after the router is frozen (AGENTS.md).
#
#   sbatch cluster/eval_v7_router.sbatch                 # dev
#   SPLIT=test sbatch cluster/eval_v7_router.sbatch
#   FORMAT_GUARD=1 sbatch cluster/eval_v7_router.sbatch      # with the format guard
#
# Needs benchmarks/mixed_ua_v7.jsonl, router/intent_rules.py and the current
# scripts/run_small_router_cluster.py on the lab (scp + checksum). Writes
# results/v7_<split>_router/. Not yet run: try SPLIT=dev first.

set -euo pipefail

SPLIT="${SPLIT:-dev}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
BENCH="${BENCH:-benchmarks/mixed_ua_v7.jsonl}"
SYSTEM="router_matrix_v2_fewshot"
# FORMAT_GUARD=1 adds router/format_guard.py (system name gets the suffix _fg).
FORMAT_GUARD="${FORMAT_GUARD:-0}"
GUARD_ARGS=()
SCORES="scores_router.json"
if [[ "${FORMAT_GUARD}" == "1" ]]; then
  GUARD_ARGS=(--format-guard)
  SYSTEM="${SYSTEM}_fg"
  SCORES="scores_router_fg.json"
fi
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
OUT_DIR="${ROOT}/results/v7_${SPLIT}_router"
mkdir -p "${ROOT}/logs" "${OUT_DIR}"
if [[ ! -f "${ROOT}/${BENCH}" ]]; then
  echo "[v7-router] missing ${BENCH}; scp it from the laptop first" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[v7-router] need 3 GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi
echo "[v7-router] host=$(hostname) gpus=${GPUS[*]} split=${SPLIT}"
nvidia-smi -L

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

serve_one() {
  local gpu="$1" model="$2" port="$3" name="$4"
  echo "[v7-router] ${name} gpu=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
    --host 127.0.0.1 --port "${port}" --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --trust-remote-code --enforce-eager &
  PIDS+=("$!")
}

serve_one "${GPUS[0]}" "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003 mamay4
serve_one "${GPUS[1]}" "lapa-llm/lapa-v0.1.2-instruct" 8001 lapa
serve_one "${GPUS[2]}" "CohereLabs/aya-expanse-8b" 8005 aya

cd "${ROOT}"
for port in 8003 8001 8005; do
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null && break
    sleep 10
  done
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null || { echo "[v7-router] TIMEOUT on :${port}" >&2; exit 1; }
done

# Rules v2 with the alignment few-shot prompt: the frozen one-hop router.
python scripts/run_small_router_cluster.py --mode router_matrix --align-prompt fewshot \
  --bench "${BENCH}" --split "${SPLIT}" --repeats 1 --max-tokens "${MAX_TOKENS}" --out-dir "${OUT_DIR}" \
  "${GUARD_ARGS[@]}"
python scripts/score_results.py "${OUT_DIR}/${SYSTEM}"_mixed_ua_v7_"${SPLIT}"_*.jsonl \
  --out "${OUT_DIR}/${SCORES}"

echo "[v7-router] finished ${SPLIT}"
