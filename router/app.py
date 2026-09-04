"""OpenAI-compatible shim + playground UI.

LiteLLM (optional, OpenAI path):
  litellm --config litellm/config.yaml --port 4000

Playground talks to vLLM ports directly (SSH tunnels to the lab):
  uvicorn router.app:app --port 4010
  open http://localhost:4010/
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from router.backends import BACKENDS, MODEL_META, PLAYGROUND_MODELS
from router.cascade import should_retry_mamay4_micro
from router.client import async_chat
from router.ensemble import (
    ENSEMBLE_VOTERS,
    extract_vote_label,
    majority_winner,
    should_ensemble_vote,
)
from router.intent_rules import ROUTER_PROFILES, extract_user_text, route_intent
from router.orchestrator import Plan, execute_plan
from router.planner import generate_plan

LITELLM_BASE = os.getenv("LITELLM_BASE", "http://localhost:4000").rstrip("/")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-diploma-dev")
AUTO_MODELS = {"auto", "router", "ua-router"}
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="UA specialist intent router", version="0.2.0")
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
def playground() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="playground UI missing")
    return FileResponse(index)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/demo/catalog")
def catalog() -> dict[str, Any]:
    return {
        "models": [
            {"id": alias, **MODEL_META[alias]} for alias in PLAYGROUND_MODELS
        ],
        "profiles": {
            key: {
                "id": key,
                "label": spec["label"],
                "blurb": spec["blurb"],
                "default": spec["default"],
            }
            for key, spec in ROUTER_PROFILES.items()
        },
        "modes": [
            {
                "id": "single",
                "label": "Single specialist",
                "blurb": "Force one model; router is bypassed.",
            },
            {
                "id": "router",
                "label": "Rules router",
                "blurb": "Regex intent → one specialist from the pool.",
            },
            {
                "id": "compare",
                "label": "Compare pool",
                "blurb": "Same prompt to every selected specialist.",
            },
            {
                "id": "cascade",
                "label": "Micro-cascade",
                "blurb": "Rules hop, then Mamay-4B retry on social 2 / missing digit.",
            },
            {
                "id": "ensemble",
                "label": "Ensemble vote",
                "blurb": "Majority of Lapa + Mamay-4B + Aya on alignment / ZNO-like items.",
            },
            {
                "id": "orchestrator",
                "label": "Composite orchestrator",
                "blurb": "Plan dependent specialist steps, execute them, return the final artifact.",
            },
        ],
    }


@app.get("/v1/demo/health")
async def demo_health() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=2.5) as client:
        for alias in PLAYGROUND_MODELS:
            base, hf = BACKENDS[alias]
            url = f"{base}/models"
            ok = False
            detail = ""
            try:
                resp = await client.get(url)
                ok = resp.status_code == 200
                detail = f"HTTP {resp.status_code}"
            except Exception as exc:  # noqa: BLE001 — surface any transport error
                detail = f"{type(exc).__name__}: {exc}"
            meta = MODEL_META[alias]
            rows.append(
                {
                    "id": alias,
                    "ok": ok,
                    "detail": detail,
                    "url": url,
                    "hf": hf,
                    **meta,
                }
            )
    return {"backends": rows, "any_up": any(r["ok"] for r in rows)}


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

        plan, planning = await generate_plan(prompt, caller)
    return {"plan": plan.to_dict(), "planning": planning}


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
            plan, planning = await generate_plan(prompt, caller)
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


async def _chat_one(
    client: httpx.AsyncClient,
    alias: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    return await async_chat(
        client,
        alias,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )


def _as_item(prompt: str, intent: str) -> dict[str, str]:
    """Minimal item shape so cascade / ensemble helpers can run on free-form chat."""
    if intent == "alignment":
        return {"bucket": "alignment", "id": "ualign-social-live", "prompt": prompt}
    if intent == "knowledge":
        return {"bucket": "knowledge", "id": "zno-live", "prompt": prompt}
    return {"bucket": intent or "chat", "id": "live", "prompt": prompt}


@app.post("/v1/demo/complete")
async def demo_complete(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    prompt = str(body.get("prompt") or extract_user_text(messages)).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")

    mode = str(body.get("mode") or "router")
    profile = str(body.get("profile") or "v2")
    pool = _pool_from_body(body) or frozenset(PLAYGROUND_MODELS)
    forced = str(body.get("model") or "mamay4")
    temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
    max_tokens = int(body.get("max_tokens") or 256)
    seed = body.get("seed")
    seed_i = int(seed) if seed is not None else 42

    decision = route_intent(prompt, profile=profile, pool=pool)
    timeout = httpx.Timeout(180.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if mode == "single":
            if forced not in BACKENDS:
                raise HTTPException(status_code=400, detail=f"unknown model {forced}")
            hop = await _chat_one(
                client, forced, prompt, temperature=temperature, max_tokens=max_tokens, seed=seed_i
            )
            return {
                "mode": mode,
                "route": {
                    "model": forced,
                    "intent": "explicit",
                    "reason": "client chose model",
                },
                "answers": [{**hop, "primary": True}],
            }

        if mode == "compare":
            aliases = [a for a in PLAYGROUND_MODELS if a in pool]
            if not aliases:
                raise HTTPException(status_code=400, detail="empty pool")
            hops = await asyncio.gather(
                *[
                    _chat_one(
                        client,
                        alias,
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed_i,
                    )
                    for alias in aliases
                ]
            )
            return {
                "mode": mode,
                "route": {
                    "model": decision.model,
                    "intent": decision.intent,
                    "reason": decision.reason,
                },
                "answers": [{**h, "primary": h["alias"] == decision.model} for h in hops],
            }

        first_alias = decision.model
        first = await _chat_one(
            client,
            first_alias,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed_i,
        )
        answers = [{**first, "primary": True}]
        extra: dict[str, Any] = {}

        if mode == "cascade":
            item = _as_item(prompt, decision.intent)
            conf = should_retry_mamay4_micro(item, first.get("content") or "")
            extra["cascade"] = {
                "kind": "micro",
                "first_model": first_alias,
                "escalated": conf.escalate,
                "confidence": conf.confidence,
                "reason": conf.reason,
            }
            if conf.escalate and "mamay4" in BACKENDS:
                retry = await _chat_one(
                    client,
                    "mamay4",
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed_i,
                )
                retry["primary"] = True
                first["primary"] = False
                answers = [first, retry]
                extra["cascade"]["final_model"] = "mamay4"
                extra["route_final"] = {
                    "model": "mamay4",
                    "intent": decision.intent,
                    "reason": f"micro cascade ← {conf.reason}",
                }

        if mode == "ensemble":
            item = _as_item(prompt, decision.intent)
            if should_ensemble_vote(item):
                ballots: list[tuple[str, str | None, dict]] = [
                    (first_alias, extract_vote_label(item, first.get("content") or ""), first)
                ]
                for voter in ENSEMBLE_VOTERS:
                    if voter == first_alias or voter not in pool:
                        continue
                    extra_hop = await _chat_one(
                        client,
                        voter,
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed_i,
                    )
                    answers.append({**extra_hop, "primary": False})
                    ballots.append(
                        (
                            voter,
                            extract_vote_label(item, extra_hop.get("content") or ""),
                            extra_hop,
                        )
                    )
                win_alias, win_lab, _ = majority_winner(ballots, first_alias)
                extra["ensemble"] = {
                    "voted": True,
                    "ballots": {a: lab for a, lab, _ in ballots},
                    "winner_label": win_lab,
                    "winner_model": win_alias,
                    "flipped": win_alias != first_alias,
                }
                for ans in answers:
                    ans["primary"] = ans["alias"] == win_alias
            else:
                extra["ensemble"] = {
                    "voted": False,
                    "reason": "not a discrete alignment / ZNO-like item; rules hop only",
                }

        return {
            "mode": mode,
            "route": {
                "model": decision.model,
                "intent": decision.intent,
                "reason": decision.reason,
            },
            "answers": answers,
            **extra,
        }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested = body.get("model") or "auto"
    profile = str(body.get("router_profile") or "v2")
    pool = _pool_from_body(body)

    if requested in AUTO_MODELS:
        text = extract_user_text(body.get("messages") or [])
        decision = route_intent(text, profile=profile, pool=pool)
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
