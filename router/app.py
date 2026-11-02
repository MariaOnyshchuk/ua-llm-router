"""OpenAI-compatible router endpoint and the demo page.

  uvicorn router.app:app --port 4010
  open http://localhost:4010/        (redirects to /debug, the one page)

The page and its API live in router/debug.py and router/static/debug.html. The specialists
are reached on their vLLM ports (SSH tunnels to the lab), listed in router/backends.py.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from router.backends import BACKENDS, PLAYGROUND_MODELS
from router.client import async_chat
from router.debug import router as debug_router
from router.intent_rules import extract_user_text, route_intent
from router.orchestrator import Plan, execute_plan
from router.planner import PLANNER_PROFILES, generate_plan
from router.wiki_search import WikiClient, wiki_augment

AUTO_MODELS = {"auto", "router", "ua-router"}
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Wikipedia passages for knowledge requests: per request with "wiki": true, or for all with ROUTER_WIKI=1.
WIKI_DEFAULT = os.getenv("ROUTER_WIKI") == "1"
_wiki_client: WikiClient | None = None


def _wiki() -> WikiClient:
    global _wiki_client
    if _wiki_client is None:
        _wiki_client = WikiClient()
    return _wiki_client


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"] if isinstance(m.get("content"), str) else ""
    return ""


def _with_last_user_text(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = text
            break
    return out

app = FastAPI(title="UA specialist intent router", version="0.2.0")
app.include_router(debug_router)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _header_safe(value: str) -> str:
    """HTTP headers must be latin-1; keep full text in JSON body instead."""
    return value.encode("ascii", "replace").decode("ascii")


def _pool_from_body(body: dict[str, Any]) -> frozenset[str] | None:
    raw = body.get("pool")
    if not raw:
        return None
    return frozenset(str(x) for x in raw if str(x) in BACKENDS)


@app.get("/")
def playground() -> RedirectResponse:
    """There is one page: the demo and debug view at /debug."""
    return RedirectResponse("/debug")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/route/preview")
async def preview(body: dict[str, Any]) -> dict[str, Any]:
    """Dry-run: show which specialist would be chosen (no LLM call)."""
    messages = body.get("messages") or []
    prompt = body.get("prompt") or extract_user_text(messages)
    profile = str(body.get("profile") or "v2")
    pool = _pool_from_body(body)
    decision = route_intent(prompt, profile=profile, pool=pool)
    return {
        "model": decision.model,
        "intent": decision.intent,
        "reason": decision.reason,
        "profile": profile,
        "preview_text": prompt[:500],
    }


@app.post("/v1/orchestrate/preview")
async def orchestrate_preview(body: dict[str, Any]) -> dict[str, Any]:
    """Generate and validate a plan without executing specialist steps."""
    prompt = str(
        body.get("prompt") or extract_user_text(body.get("messages") or [])
    ).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    temperature = float(body.get("temperature", 0.0))
    max_tokens = int(body.get("max_tokens") or 512)
    seed = int(body.get("seed", 42))
    planner_profile = str(body.get("planner_profile") or "hybrid")
    if planner_profile not in PLANNER_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"planner_profile must be one of {sorted(PLANNER_PROFILES)}",
        )
    async with httpx.AsyncClient(timeout=180.0) as client:
        async def caller(alias: str, step_prompt: str) -> dict[str, Any]:
            return await async_chat(
                client,
                alias,
                step_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )

        plan, planning = await generate_plan(prompt, caller, profile=planner_profile)
    return {"plan": plan.to_dict(), "planning": planning, "planner_profile": planner_profile}


@app.post("/v1/orchestrate")
async def orchestrate(body: dict[str, Any]) -> dict[str, Any]:
    """Plan and execute a bounded dependent-specialist workflow."""
    prompt = str(
        body.get("prompt") or extract_user_text(body.get("messages") or [])
    ).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    temperature = float(body.get("temperature", 0.0))
    max_tokens = int(body.get("max_tokens") or 512)
    seed = int(body.get("seed", 42))
    profile = str(body.get("profile") or "v2")
    planner_profile = str(body.get("planner_profile") or "hybrid")
    if planner_profile not in PLANNER_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"planner_profile must be one of {sorted(PLANNER_PROFILES)}",
        )
    pool = _pool_from_body(body) or frozenset(PLAYGROUND_MODELS)

    async with httpx.AsyncClient(timeout=180.0) as client:
        async def caller(alias: str, step_prompt: str) -> dict[str, Any]:
            return await async_chat(
                client,
                alias,
                step_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )

        oracle_raw = body.get("oracle_plan")
        if oracle_raw:
            plan = Plan.from_dict(oracle_raw, source="provided_oracle")
            planning: dict[str, Any] = {
                "valid": True,
                "repaired": False,
                "attempts": [],
            }
        else:
            plan, planning = await generate_plan(
                prompt, caller, profile=planner_profile
            )
        execution = await execute_plan(
            plan,
            caller,
            profile=profile,
            pool=pool,
        )

    planner_attempts = planning.get("attempts") or []
    planner_latency = sum(float(x.get("latency_ms") or 0) for x in planner_attempts)
    planner_gpu = sum(float(x.get("gpu_seconds") or 0) for x in planner_attempts)
    return {
        "mode": "orchestrator",
        "planning": planning,
        **execution,
        "calls": int(execution["calls"]) + len(planner_attempts),
        "planner_calls": len(planner_attempts),
        "latency_ms": round(float(execution["latency_ms"]) + planner_latency, 1),
        "gpu_seconds": round(float(execution["gpu_seconds"]) + planner_gpu, 4),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested = body.get("model") or "auto"
    profile = str(body.get("router_profile") or "v2")
    pool = _pool_from_body(body)
    use_wiki = bool(body.pop("wiki", WIKI_DEFAULT))

    if requested in AUTO_MODELS:
        text = extract_user_text(body.get("messages") or [])
        decision = route_intent(text, profile=profile, pool=pool)
        body = {**body, "model": decision.model}
        wiki_meta = None
        user_text = _last_user_text(body.get("messages") or [])
        if use_wiki and user_text:
            augmented, wiki_meta = await asyncio.to_thread(wiki_augment, user_text, decision.intent, _wiki())
            if augmented != user_text:
                body["messages"] = _with_last_user_text(body.get("messages") or [], augmented)
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
        if wiki_meta is not None:
            route_body["wiki"] = {
                "used": wiki_meta.get("used"),
                "gate": wiki_meta.get("gate"),
                "queries": wiki_meta.get("queries"),
                "sources": [
                    {"title": x["title"], "url": x["url"], "score": x["score"]}
                    for x in wiki_meta.get("passages") or []
                ],
            }
            route_meta["x-router-wiki"] = str(len(route_body["wiki"]["sources"]))
    else:
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
    alias = str(body.get("model"))
    if alias not in BACKENDS:
        raise HTTPException(status_code=400, detail=f"unknown model {alias}; use one of {sorted(BACKENDS)} or auto")
    # the router alias (e.g. "lapa") becomes the vLLM base URL and the served model name
    base, served_model = BACKENDS[alias]
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = {**body, "model": served_model}
    body.pop("pool", None)
    body.pop("router_profile", None)

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
        data["router"] = route_body
        return JSONResponse(data, headers=route_meta)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "owned_by": "router"},
            {"id": "mamay4", "object": "model", "owned_by": "specialist"},
            {"id": "lapa", "object": "model", "owned_by": "specialist"},
            {"id": "aya", "object": "model", "owned_by": "specialist"},
            {"id": "qwen7", "object": "model", "owned_by": "specialist"},
            {"id": "mamay12", "object": "model", "owned_by": "specialist"},
            {"id": "qwen", "object": "model", "owned_by": "specialist"},
        ],
    }
