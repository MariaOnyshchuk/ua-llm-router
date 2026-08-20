#!/bin/bash
# Create isolated venv + install vLLM on ucu-lab-2240.
# Run on the cluster: bash cluster/setup_env.sh
set -euo pipefail

ROOT="${HOME}/Diploma"
VENV="${ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

mkdir -p "${ROOT}/logs" "${ROOT}/cluster"
cd "${ROOT}"

if [[ ! -d "${VENV}" ]]; then
  echo "[setup] creating venv at ${VENV}"
  "${PYTHON}" -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

echo "[setup] installing vllm (this can take several minutes)..."
# Pin lightly; bump later if needed
pip install "vllm>=0.8.0" huggingface_hub

python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
echo "[setup] done."
