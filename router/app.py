"""Tiny OpenAI-compatible shim: model=auto → rules → LiteLLM.

Run LiteLLM first:
  litellm --config litellm/config.yaml --port 4000

Then:
  pip install -r requirements.txt
  uvicorn router.app:app --reload --port 4010

Client:
  base_url=http://localhost:4010/v1  model=auto
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router.intent_rules import extract_user_text, route_intent

LITELLM_BASE = os.getenv("LITELLM_BASE", "http://localhost:4000").rstrip("/")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-diploma-dev")
AUTO_MODELS = {"auto", "router", "ua-router"}

app = FastAPI(title="UA specialist intent router", version="0.1.0")


def _header_safe(value: str) -> str:
    """HTTP headers must be latin-1; keep full text in JSON body instead."""
    return value.encode("ascii", "replace").decode("ascii")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/route/preview")
async def preview(body: dict[str, Any]) -> dict[str, Any]:
    """Dry-run: show which specialist would be chosen (no LLM call)."""
    messages = body.get("messages") or []
    prompt = body.get("prompt") or extract_user_text(messages)
    decision = route_intent(prompt)
    return {
        "model": decision.model,
        "intent": decision.intent,
        "reason": decision.reason,
        "preview_text": prompt[:500],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested = body.get("model") or "auto"

    if requested in AUTO_MODELS:
        text = extract_user_text(body.get("messages") or [])
        decision = route_intent(text)
        body = {**body, "model": decision.model}
        route_meta = {
            "x-router-model": _header_safe(decision.model),
            "x-router-intent": _header_safe(decision.intent),
            "x-router-reason": _header_safe(decision.reason),
        }
        route_body = {
            "model": decision.model,
            "intent": decision.intent,
            "reason": decision.reason,
        }
    else:
        # Explicit alias: pass through (lapa|mamay|gemma|qwen|…)
        route_meta = {
            "x-router-model": _header_safe(requested),
            "x-router-intent": "explicit",
            "x-router-reason": "client chose model",
        }
        route_body = {
            "model": requested,
            "intent": "explicit",
            "reason": "client chose model",
        }

    stream = bool(body.get("stream"))
    headers = {
        "Authorization": f"Bearer {LITELLM_KEY}",
        "Content-Type": "application/json",
    }

    url = f"{LITELLM_BASE}/v1/chat/completions"

    if stream:
        client = httpx.AsyncClient(timeout=None)

        async def upstream():
            try:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code >= 400:
                        err = await resp.aread()
                        yield err
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(
            upstream(),
            media_type="text/event-stream",
            headers=route_meta,
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        # Surface routing choice in the OpenAI-style payload for debugging
        data["router"] = route_body
        return JSONResponse(data, headers=route_meta)


@app.get("/v1/models")
async def list_models():
    # Matrix router targets (week-4 bake-off): mamay4 | mamay12 | lapa
    return {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "owned_by": "router"},
            {"id": "mamay4", "object": "model", "owned_by": "specialist"},
            {"id": "lapa", "object": "model", "owned_by": "specialist"},
            {"id": "aya", "object": "model", "owned_by": "specialist"},
            {"id": "qwen7", "object": "model", "owned_by": "specialist"},
            {"id": "mamay12", "object": "model", "owned_by": "specialist"},
            {"id": "qwen", "object": "model", "owned_by": "specialist"},  # legacy ablation
        ],
    }
