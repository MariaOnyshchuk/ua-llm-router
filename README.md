# UA Specialist Router (LiteLLM)

Short design: [DESIGN.md](DESIGN.md)

## Quick start (routing rules only)

```bash
cd Diploma
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/smoke_test.sh
```

## Full stack (when vLLM backends exist)

```bash
# Terminal A — LiteLLM
litellm --config litellm/config.yaml --port 4000

# Terminal B — rules router
uvicorn router.app:app --port 4010

# Client
curl http://localhost:4010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role":"user","content":"Переклади: Hello world"}]
  }'
```

Aliases: `auto` | `lapa` | `mamay` | `gemma` | `qwen`
