#!/bin/bash
#SBATCH --job-name=mamay27-ref
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/Diploma/logs/mamay27-ref-%j.out
#SBATCH --error=/home/%u/Diploma/logs/mamay27-ref-%j.err

# Mamay-27B as the reference model, on the lab (4x RTX 6000 Ada, 48 GB each).
# Numbers from one machine are compared only with numbers from the same machine, so the three router
# models are run here too, on the same sets:
#   part 1: Mamay-4B, Lapa, Aya in bf16, one card each (gpu_memory_utilization 0.90);
#   part 2: Mamay-27B in bf16 on two cards (tensor parallel 2; about 54 GB of weights does not fit one
#           48 GB card), alone on the machine's free cards.
# T=0, seed=42, max_tokens=1024, one run. Sets: nodes_mixed_600_test and graphs_mixed_100_dev, every model
# answering each item in one call. Raw rows go to logs/<set>_lab, scores to results/<set>_lab.
# A card that holds another user's process is skipped (the job asks Slurm for all four).
#
#   sbatch cluster/eval_mamay27_reference_lab.sbatch
#   BENCHES="graphs_mixed_100_dev" PARTS="27b" sbatch --export=ALL cluster/eval_mamay27_reference_lab.sbatch

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

BENCHES="${BENCHES:-nodes_mixed_600_test graphs_mixed_100_dev}"
PARTS="${PARTS:-small 27b}"
cd "${ROOT}"
mkdir -p logs

# Free cards: less than 1 GB in use.
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 1024 {print $1}')
echo "[ref] host=$(hostname) free cards=${FREE[*]} benches=${BENCHES} parts=${PARTS}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
if [[ "${#FREE[@]}" -lt 3 ]]; then
  echo "[ref] need 3 free cards, have ${#FREE[@]}" >&2
  exit 1
fi

PIDS=()
stop_all() {
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
  PIDS=()
}
trap stop_all EXIT

serve() {
  local gpus="$1" model="$2" port="$3" name="$4" tp="$5"
  echo "[ref] serve ${name} cards=${gpus} port=${port} tp=${tp}"
  CUDA_VISIBLE_DEVICES="${gpus}" vllm serve "${model}" \
    --host 127.0.0.1 --port "${port}" \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size "${tp}" \
    --trust-remote-code --enforce-eager > "logs/serve-${name}-${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=("$!")
}

wait_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[ref] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[ref] TIMEOUT ${name} :${port}" >&2
  tail -40 "logs/serve-${name}-${SLURM_JOB_ID}.log" >&2
  return 1
}

run_set() {
  local system="$1" bench="$2"
  local raw="logs/${bench}_lab" out="results/${bench}_lab"
  mkdir -p "${raw}" "${out}"
  python scripts/run_composite_router.py \
    --bench "benchmarks/${bench}.jsonl" --systems "${system}" \
    --repeats 1 --temperature 0 --seed 42 --max-tokens 1024 \
    --out-dir "${raw}" > "logs/ref-run-${system}-${bench}-${SLURM_JOB_ID}.log" 2>&1
}

score_set() {
  local system="$1" bench="$2"
  local raw="logs/${bench}_lab" out="results/${bench}_lab"
  local newest
  newest=$(ls -t "${raw}/${system}_composite_${bench}_"*.jsonl | head -1)
  python scripts/score_composite_results.py "${newest}" --out "${out}/scores_${system}.json"
  echo "[ref] ${system} ${bench} -> ${out}/scores_${system}.json"
}

for bench in ${BENCHES}; do sha256sum "benchmarks/${bench}.jsonl"; done

if [[ " ${PARTS} " == *" small "* ]]; then
  serve "${FREE[0]}" "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003 mamay4 1
  serve "${FREE[1]}" "lapa-llm/lapa-v0.1.2-instruct" 8001 lapa 1
  serve "${FREE[2]}" "CohereLabs/aya-expanse-8b" 8005 aya 1
  wait_port 8003 mamay4; wait_port 8001 lapa; wait_port 8005 aya
  nvidia-smi --query-gpu=index,memory.used --format=csv
  for bench in ${BENCHES}; do
    RUNNERS=()
    for system in mamay4 lapa aya; do
      run_set "${system}" "${bench}" &
      RUNNERS+=("$!")
    done
    status=0
    for pid in "${RUNNERS[@]}"; do wait "${pid}" || status=1; done
    [[ "${status}" == 0 ]] || { echo "[ref] a runner failed on ${bench}" >&2; exit 1; }
    for system in mamay4 lapa aya; do score_set "${system}" "${bench}"; done
  done
  stop_all
  sleep 20
fi

if [[ " ${PARTS} " == *" 27b "* ]]; then
  serve "${FREE[0]},${FREE[1]}" "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0" 8002 mamay27 2
  wait_port 8002 mamay27
  nvidia-smi --query-gpu=index,memory.used --format=csv
  for bench in ${BENCHES}; do
    run_set mamay27 "${bench}"
    score_set mamay27 "${bench}"
  done
  stop_all
fi

echo "[ref] finished"
