#!/bin/bash
#SBATCH --job-name=comp-v2-bake
#SBATCH --partition=gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/Diploma/logs/comp-v2-bake-%j.out
#SBATCH --error=/home/%u/Diploma/logs/comp-v2-bake-%j.err

# Held-out composite v2 bake-off on the frozen TEST split only.
# Systems: router_direct, oracle, hybrid (free-form baseline), template (frozen).
# Decode: T=0, seed=42, max_tokens=256, 3 repeats.
#
#   sbatch cluster/eval_composite_v2_bakeoff.sbatch
#
# Optional env:
#   SPLIT=test          (default; do not use for tuning)
#   SYSTEMS=router_direct,oracle,hybrid,template
#   REPEATS=3

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

SPLIT="${SPLIT:-test}"
SYSTEMS="${SYSTEMS:-router_direct,oracle,hybrid,template}"
REPEATS="${REPEATS:-3}"
OUT_DIR="${ROOT}/results/week_12_composite_v2_test_bakeoff"

mkdir -p "${ROOT}/logs" "${OUT_DIR}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[comp-v2] need 3 GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

echo "[comp-v2] host=$(hostname) gpus=${GPUS[*]} split=${SPLIT} systems=${SYSTEMS}"
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
  echo "[comp-v2] ${name} gpu=${gpu} port=${port}"
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
      echo "[comp-v2] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[comp-v2] TIMEOUT waiting for ${name} on :${port}" >&2
  return 1
}

wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya

cd "${ROOT}"
python scripts/build_composite_benchmark.py --version v2 --check

python scripts/run_composite_router.py \
  --bench benchmarks/mixed_ua_composite_v2.jsonl \
  --systems "${SYSTEMS}" \
  --split "${SPLIT}" \
  --repeats "${REPEATS}" \
  --temperature 0 \
  --seed 42 \
  --max-tokens 256 \
  --out-dir "${OUT_DIR}"

# Score each system separately (de-dupe safety), then a combined summary.
python - <<'PY'
from pathlib import Path
import subprocess, sys
out = Path("results/week_12_composite_v2_test_bakeoff")
files = sorted(out.glob("*_composite_v2_*.jsonl"))
# Drop empty / incomplete: require HTTP-complete via scorer http_ok_rate later.
if not files:
    raise SystemExit("no bake-off jsonl files")
subprocess.check_call(
    [sys.executable, "scripts/score_composite_results.py", *[str(p) for p in files],
     "--out", str(out / "scores.json")]
)
print("scored", len(files), "files")
PY

echo "[comp-v2] finished → ${OUT_DIR}/scores.json"
