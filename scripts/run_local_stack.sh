#!/usr/bin/env bash
# End-to-end local stack against Lapa on 2240 (via tunnel).
# Usage: bash scripts/run_local_stack.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1) Ensure SSH tunnel to vLLM
if ! curl -sf -o /dev/null http://127.0.0.1:8001/v1/models; then
  echo "[stack] opening tunnel localhost:8001 -> ucu-lab-2240:8001"
  lsof -ti:8001 | xargs kill -9 2>/dev/null || true
  ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -L 8001:localhost:8001 ucu-lab-2240
  sleep 1
fi
curl -sf http://127.0.0.1:8001/v1/models >/dev/null
echo "[stack] Lapa reachable on :8001"

# 2) venv + deps
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

# 3) LiteLLM on :4000
lsof -ti:4000 | xargs kill -9 2>/dev/null || true
echo "[stack] starting LiteLLM on :4000"
nohup litellm --config litellm/config.yaml --port 4000 > logs/litellm.log 2>&1 &
echo $! > logs/litellm.pid
sleep 3

# 4) Rules router on :4010
lsof -ti:4010 | xargs kill -9 2>/dev/null || true
echo "[stack] starting intent router on :4010"
mkdir -p logs
nohup env PYTHONPATH="$ROOT" uvicorn router.app:app --host 127.0.0.1 --port 4010 > logs/router.log 2>&1 &
echo $! > logs/router.pid
sleep 2

echo "[stack] ready"
echo "  preview:  curl -s localhost:4010/v1/route/preview -H 'Content-Type: application/json' -d '{\"prompt\":\"Переклади hello\"}'"
echo "  chat:     curl -s localhost:4010/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"Переклади: hello\"}],\"max_tokens\":64}'"
