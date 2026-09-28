#!/bin/bash
#SBATCH --job-name=agentcoma200
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/Diploma/logs/agentcoma200-%j.out
#SBATCH --error=/home/%u/Diploma/logs/agentcoma200-%j.err

# AgentCoMa-UK 200 (50 dev + 150 test) end-task run with the composite runner.
# Systems: router_direct, oracle, hybrid, template. Decode: T=0, seed=42,
# max_tokens=256, 3 repeats. Default SPLIT=dev for tuning; run SPLIT=test once.
#
#   sbatch cluster/eval_agentcoma_200.sbatch
#   SPLIT=test sbatch cluster/eval_agentcoma_200.sbatch
#
# The source files benchmarks/agentcoma_uk_200_merged.jsonl and
# benchmarks/agentcoma_uk_200_composite.jsonl are NOT in git: scp them to the
# lab first. Generation JSONL stays on the lab / HF; only scores come back.
#
# Optional env: SPLIT=dev|test  SYSTEMS=...  REPEATS=3

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

SPLIT="${SPLIT:-dev}"
SYSTEMS="${SYSTEMS:-router_direct,oracle,hybrid,template}"
REPEATS="${REPEATS:-3}"
OUT_DIR="${ROOT}/results/week_13_agentcoma_200_${SPLIT}"
BENCH="${ROOT}/benchmarks/agentcoma_uk_200_composite.jsonl"

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

if [[ ! -f "${BENCH}" ]]; then
  echo "[agentcoma200] missing ${BENCH}; scp it from the laptop first" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[agentcoma200] need 3 GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

echo "[agentcoma200] host=$(hostname) gpus=${GPUS[*]} split=${SPLIT} systems=${SYSTEMS}"
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
  echo "[agentcoma200] ${name} gpu=${gpu} port=${port}"
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
      echo "[agentcoma200] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[agentcoma200] TIMEOUT waiting for ${name} on :${port}" >&2
  return 1
}

wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya

cd "${ROOT}"

python scripts/run_composite_router.py \
  --bench "${BENCH}" \
  --systems "${SYSTEMS}" \
  --split "${SPLIT}" \
  --repeats "${REPEATS}" \
  --temperature 0 \
  --seed 42 \
  --max-tokens 256 \
  --out-dir "${OUT_DIR}"

python - "${OUT_DIR}" <<'PY'
from pathlib import Path
import subprocess, sys
out = Path(sys.argv[1])
files = sorted(p for p in out.glob("*_composite_agentcoma200_*.jsonl"))
if not files:
    raise SystemExit("no agentcoma200 jsonl files")
subprocess.check_call(
    [sys.executable, "scripts/score_composite_results.py", *[str(p) for p in files],
     "--out", str(out / "scores.json")]
)
print("scored", len(files), "files")
PY

echo "[agentcoma200] finished → ${OUT_DIR}/scores.json"
