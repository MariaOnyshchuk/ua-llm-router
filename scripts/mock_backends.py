#!/usr/bin/env python3
"""Fake OpenAI-compatible specialists for the debug UI (no GPU, no tunnel).

Listens on the ports in router/backends.py (mamay4 8003, lapa 8001, aya 8005, mamay12 8002,
qwen7 8004) and streams a canned answer word by word. Answers are NOT model outputs:
letters and digits come from a hash of the prompt, so scores in mock mode mean nothing.

    python scripts/mock_backends.py
    uvicorn router.app:app --port 4010     # then open http://localhost:4010/debug
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.backends import BACKENDS  # noqa: E402


def canned(alias: str, prompt: str) -> str:
    h = int(hashlib.sha1(prompt.encode()).hexdigest(), 16)
    if "лише літера" in prompt or "літера варіанту" in prompt:
        return f"Відповідь: {'ABCD'[h % 4]}"
    if "0, 1 або 2" in prompt:
        return str(h % 3)
    return f"[mock {alias}] Це тестова відповідь для налагодження інтерфейсу. Відповідь: {h % 20}"


def make_app(alias: str) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": BACKENDS[alias][1], "mock": True}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        prompt = body["messages"][-1]["content"]
        text = canned(alias, prompt)
        words = text.split(" ")
        if not body.get("stream"):
            return JSONResponse({
                "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(words),
                          "total_tokens": len(prompt.split()) + len(words)},
                "model": body.get("model"),
            })

        async def gen():
            for i, w in enumerate(words):
                await asyncio.sleep(0.03)
                piece = w if i == 0 else " " + w
                yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                yield "data: " + json.dumps({"choices": [], "usage": {
                    "prompt_tokens": len(prompt.split()), "completion_tokens": len(words),
                    "total_tokens": len(prompt.split()) + len(words)}}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main() -> None:
    ports: dict[int, str] = {}
    for alias, (base, _model) in BACKENDS.items():
        ports.setdefault(urlparse(base).port, alias)  # mamay12 and mamay27 share 8002
    servers = []
    for port, alias in ports.items():
        cfg = uvicorn.Config(make_app(alias), host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        server.install_signal_handlers = lambda: None
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        print(f"mock {alias:8s} on 127.0.0.1:{port}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        for s in servers:
            s.should_exit = True


if __name__ == "__main__":
    main()
