#!/bin/bash
#SBATCH --job-name=planners
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# The lab refuses jobs longer than 6 h ("InvalidAccount"); rerun what is left with SYSTEMS=...
#SBATCH --output=/home/%u/Diploma/logs/planners-%j.out
#SBATCH --error=/home/%u/Diploma/logs/planners-%j.err

# Planner comparison on graphs_mixed_100_dev (dev, for tuning): score and latency of the whole request,
# planner call(s) included, on the lab (4x RTX 6000 Ada), so it compares with the lab single-model runs of
# cluster/eval_mamay27_reference_lab.sbatch. Gate-free planner and the prompts of commit f221521 or later.
#
# Mamay-4B, Lapa and Aya in bf16, one card each, gpu_memory_utilization 0.90; Mamay-12B on a fourth card
# when one is free (it is then used only as a planner). T=0, seed=42, max_tokens=1024, one run.
# Systems run one after another (the runner sends one request at a time), so latencies are not shared.
# The runner itself sees no GPU: the e5 encoder of the learned routers runs on the CPU and its time is
# part of the latency.
#
#   sbatch --dependency=afterany:<job> cluster/eval_planners_lab.sbatch
#   SYSTEMS="hybrid_gated hybrid_gated_aya" sbatch --export=ALL cluster/eval_planners_lab.sbatch

set -euo pipefail

ROOT="${HOME}/Diploma"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

BENCH="${BENCH:-graphs_mixed_100_dev}"
# One-call baselines, then the planner prompts (planner Mamay-4B, rules v2 routing), then other planner
# models, then the gated planner with the learned step routers.
SYSTEMS="${SYSTEMS:-router_direct router_knn_nodes_only router_mlp_nodes_only \
hybrid hybrid_minimal hybrid_fewshot hybrid_gated hybrid_gated_lapa hybrid_gated_aya \
hybrid_gated+knn_nodes_only hybrid_gated+mlp_nodes_only}"
SYSTEMS_MAMAY12="${SYSTEMS_MAMAY12:-hybrid_mamay12 hybrid_gated_mamay12}"
cd "${ROOT}"
mkdir -p logs

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 1024 {print $1}')
echo "[planners] host=$(hostname) free cards=${FREE[*]} bench=${BENCH}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
if [[ "${#FREE[@]}" -lt 3 ]]; then
  echo "[planners] need 3 free cards, have ${#FREE[@]}" >&2
  exit 1
fi
sha256sum "benchmarks/${BENCH}.jsonl" router/planner.py router/intent_rules.py

PIDS=()
stop_all() {
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap stop_all EXIT

serve() {
  local gpu="$1" model="$2" port="$3" name="$4"
  echo "[planners] serve ${name} card=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
    --host 127.0.0.1 --port "${port}" \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code --enforce-eager > "logs/serve-${name}-${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=("$!")
}

wait_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[planners] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[planners] TIMEOUT ${name} :${port}" >&2
  tail -40 "logs/serve-${name}-${SLURM_JOB_ID}.log" >&2
  return 1
}

serve "${FREE[0]}" "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003 mamay4
serve "${FREE[1]}" "lapa-llm/lapa-v0.1.2-instruct" 8001 lapa
serve "${FREE[2]}" "CohereLabs/aya-expanse-8b" 8005 aya
if [[ "${#FREE[@]}" -ge 4 ]]; then
  serve "${FREE[3]}" "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0" 8002 mamay12
  SYSTEMS="${SYSTEMS} ${SYSTEMS_MAMAY12}"
  wait_port 8002 mamay12
else
  echo "[planners] no fourth free card: skipping ${SYSTEMS_MAMAY12}"
fi
wait_port 8003 mamay4; wait_port 8001 lapa; wait_port 8005 aya
nvidia-smi --query-gpu=index,memory.used --format=csv

# Mamay-4B as the intent classifier (routing profile `llm`) on the node test and graph-dev: one short call
# per item, ids, intents and latencies only. Runs while nothing else uses the servers.
for b in nodes_mixed_600_test graphs_mixed_100_dev; do
  CUDA_VISIBLE_DEVICES="" python scripts/classify_intents_llm.py --bench "${b}" --out-dir "results/${b}_lab" \
    || echo "[planners] intent classification on ${b} had failures" >&2
done

RAW="logs/${BENCH}_lab"
OUT="results/${BENCH}_lab"
mkdir -p "${RAW}" "${OUT}"
for system in ${SYSTEMS}; do
  echo "[planners] run ${system} $(date -u +%H:%M:%S)"
  if ! CUDA_VISIBLE_DEVICES="" python scripts/run_composite_router.py \
      --bench "benchmarks/${BENCH}.jsonl" --systems "${system}" \
      --repeats 1 --temperature 0 --seed 42 --max-tokens 1024 \
      --out-dir "${RAW}" > "logs/planners-run-${system}-${SLURM_JOB_ID}.log" 2>&1; then
    echo "[planners] ${system} failed, see logs/planners-run-${system}-${SLURM_JOB_ID}.log" >&2
    continue
  fi
  newest=$(ls -t "${RAW}/${system}_composite_${BENCH}_"*.jsonl | head -1)
  python scripts/score_composite_results.py "${newest}" --out "${OUT}/scores_${system}.json"
done

echo "[planners] finished"
