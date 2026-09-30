#!/bin/bash
#SBATCH --job-name=v6-router
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/Diploma/logs/v6-router-%j.out
#SBATCH --error=/home/%u/Diploma/logs/v6-router-%j.err

# Rules-v2 router (and optionally the learned routers) on the frozen mixed_ua_v6
# suite (2414 items, 1 pass, T=0, seed=42, max_tokens=1024). Serves the three
# specialists on three GPUs (bf16, gpu_memory_utilization 0.90) and runs a phase
# of scripts/week_v6_eval_pipeline.sh.
#
#   sbatch cluster/eval_v6_router.sbatch                        # v6_rules
#   PHASE=v6_learned sbatch cluster/eval_v6_router.sbatch       # kNN and classifier routers
#
# Freeze the routing table BEFORE running this. v6 is the claim suite: do not tune on it.

set -euo pipefail

PHASE="${PHASE:-v6_rules}"
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
mkdir -p "${ROOT}/logs"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[v6-router] need 3 GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi
echo "[v6-router] host=$(hostname) gpus=${GPUS[*]} phase=${PHASE}"
nvidia-smi -L

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

serve_one() {
  local gpu="$1" model="$2" port="$3" name="$4"
  echo "[v6-router] ${name} gpu=${gpu} port=${port}"
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
# the pipeline waits for /v1/models on all three ports before it starts
bash scripts/week_v6_eval_pipeline.sh "${PHASE}"

echo "[v6-router] finished ${PHASE}"
