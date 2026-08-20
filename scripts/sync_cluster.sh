#!/usr/bin/env bash
# Copy cluster scripts to ucu-lab-2240 and (optionally) run setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-ucu-lab-2240}"

rsync -avz \
  "${ROOT}/cluster/" \
  "${ROOT}/DESIGN.md" \
  "${ROOT}/README.md" \
  "${HOST}:Diploma/" \
  --relative 2>/dev/null || true

# Explicit copy (more reliable through ProxyJump)
ssh "${HOST}" 'mkdir -p ~/Diploma/cluster ~/Diploma/logs'
scp -o BatchMode=yes \
  "${ROOT}/cluster/setup_env.sh" \
  "${ROOT}/cluster/serve_lapa.sbatch" \
  "${HOST}:Diploma/cluster/"

ssh "${HOST}" 'chmod +x ~/Diploma/cluster/setup_env.sh'
echo "Synced to ${HOST}:~/Diploma/cluster"
echo "Next on cluster: bash ~/Diploma/cluster/setup_env.sh"
