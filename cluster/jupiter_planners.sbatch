#!/bin/bash
#SBATCH --job-name=planners
#SBATCH --account=e-dev-2026d09-051
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/planners-%j.out
#SBATCH --error=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma/logs/planners-%j.err

# Planner comparison on graphs_mixed_100_dev (dev, tuning only) on JUPITER (4x GH200 96 GB).
# Planner model Mamay-4B for every planner:
#   hybrid            default prompt (SYSTEM_INSTRUCTION)
#   hybrid_fewshot    few-shot prompt
#   hybrid_gated      prompt with the node definition, may answer "one step"
#   hgpt              HuggingGPT-style: task list with <resource>-k, parallel tasks, response generation
#   compiler          LLMCompiler-style: numbered actions with $k, parallel execution, joiner (one replan)
# Steps are routed by ROUTING: v2 (intent -> model table), mlp_nodes_only or mlp_deep_nodes_only.
# The 8 demonstration items of hgpt/compiler (router/paper_planners.DEMO_IDS) are left out for every
# system (--hold-out-demos), so each run has the same 92 items.
# Mamay-4B, Lapa, Aya in bf16, one per card, gpu_memory_utilization 0.90; the runner and its e5 encoder
# use the fourth card. T=0, seed=42, max_tokens=1024, one run, one request at a time.
# Numbers from this machine are compared only with runs from this machine.
#
#   ROUTING=v2 sbatch --export=ALL cluster/jupiter_planners.sbatch
#   ROUTING=mlp_nodes_only sbatch --export=ALL cluster/jupiter_planners.sbatch
#   ROUTING=mlp_deep_nodes_only sbatch --export=ALL cluster/jupiter_planners.sbatch
#
# Mamay-27B as the planner (9 Nov 2026): MAMAY27=1 also serves Mamay-27B (~54 GB, one GH200 card) on the
# fourth card, and the runner then keeps e5 on the CPU. System `mamay27` is Mamay-27B alone, one call.
#   MAMAY27=1 ROUTING=v2 PLANNERS="mamay27 hybrid_gated_mamay27 hgpt_mamay27 compiler_mamay27" sbatch --export=ALL cluster/jupiter_planners.sbatch
#   MAMAY27=1 ROUTING=mlp_nodes_only PLANNERS="hybrid_gated_mamay27 hgpt_mamay27 compiler_mamay27" sbatch --export=ALL cluster/jupiter_planners.sbatch
#
# Planner v2 (9 Nov 2026, no demos, all 100 items):
#   MAMAY27=1 HOLD_OUT=0 ROUTING=v2 PLANNERS="planner_v2 planner_v2_mamay27" sbatch --export=ALL cluster/jupiter_planners.sbatch
#   MAMAY27=1 HOLD_OUT=0 ROUTING=mlp_nodes_only PLANNERS="planner_v2 planner_v2_mamay27" sbatch --export=ALL cluster/jupiter_planners.sbatch

set -euo pipefail

ROOT=/e/project1/e-dev-2026d09-051/onyshchuk1/Diploma
module load Stages/2026 GCCcore/14.3.0 Python/3.13.5
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_HOME=/e/project1/e-dev-2026d09-051/onyshchuk1/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

ROUTING="${ROUTING:-v2}"
MAMAY27="${MAMAY27:-0}"
# HOLD_OUT=1 leaves out the 8 demo items of hgpt/compiler (92 items). Planner v2 has no demos: HOLD_OUT=0 runs
# all 100 and the 92 are taken at analysis time.
HOLD_OUT="${HOLD_OUT:-1}"
HOLD_FLAG=""
if [[ "${HOLD_OUT}" == "1" ]]; then HOLD_FLAG="--hold-out-demos"; fi
PLANNERS="${PLANNERS:-hybrid hybrid_fewshot hybrid_gated hgpt compiler}"
BENCH=graphs_mixed_100_dev
RAW_DIR="${ROOT}/logs/${BENCH}_planners"
OUT_DIR="${ROOT}/results/${BENCH}_planners"
mkdir -p "${RAW_DIR}" "${OUT_DIR}"

echo "[planners] host=$(hostname) routing=${ROUTING} planners=${PLANNERS}"
nvidia-smi -L

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

serve_one() {
  local gpu="$1" model="$2" port="$3" name="$4"
  echo "[planners] ${name} gpu=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
    --host 127.0.0.1 --port "${port}" \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code --enforce-eager > "${ROOT}/logs/serve-${name}-${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=("$!")
}

wait_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[planners] health 200 ${name} :${port}"
      return 0
    fi
    sleep 10
  done
  echo "[planners] TIMEOUT waiting for ${name} on :${port}" >&2
  tail -30 "${ROOT}/logs/serve-${name}-${SLURM_JOB_ID}.log" >&2
  return 1
}

serve_one 0 "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0" 8003 mamay4
serve_one 1 "lapa-llm/lapa-v0.1.2-instruct" 8001 lapa
serve_one 2 "CohereLabs/aya-expanse-8b" 8005 aya
RUNNER_GPU=3
if [[ "${MAMAY27}" == "1" ]]; then
  serve_one 3 "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0" 8002 mamay27
  RUNNER_GPU=""
fi
wait_port 8003 mamay4
wait_port 8001 lapa
wait_port 8005 aya
if [[ "${MAMAY27}" == "1" ]]; then
  wait_port 8002 mamay27
fi

cd "${ROOT}"
sha256sum "benchmarks/${BENCH}.jsonl" router/planner.py router/paper_planners.py router/intent_rules.py

for planner in ${PLANNERS}; do
  system="${planner}"
  if [[ "${ROUTING}" != "v2" && "${planner}" != "mamay27" ]]; then
    system="${planner}+${ROUTING}"
  fi
  echo "[planners] run ${system} $(date -u +%H:%M:%S)"
  if ! CUDA_VISIBLE_DEVICES="${RUNNER_GPU}" python scripts/run_composite_router.py \
      --bench "benchmarks/${BENCH}.jsonl" --systems "${system}" ${HOLD_FLAG} \
      --repeats 1 --temperature 0 --seed 42 --max-tokens 1024 \
      --out-dir "${RAW_DIR}"; then
    echo "[planners] ${system} failed" >&2
    continue
  fi
  newest=$(ls -t "${RAW_DIR}/${system}_composite_${BENCH}_"*.jsonl | head -1)
  python scripts/score_composite_results.py "${newest}" --out "${OUT_DIR}/scores_${system}.json"
done

echo "[planners] finished"
