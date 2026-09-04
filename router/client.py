"""Shared OpenAI-compatible backend calls for the API and eval runners."""

from __future__ import annotations

import time
from typing import Any

import httpx

from router.backends import BACKENDS


def _payload(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    return body


def _result(alias: str, model: str, resp: httpx.Response, latency_ms: float) -> dict[str, Any]:
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    content = ""
    if resp.is_success:
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return {
        "alias": alias,
        "ok": resp.is_success,
        "content": content,
        "latency_ms": round(latency_ms, 1),
        "gpu_seconds": round(latency_ms / 1000.0, 4),
        "http_status": resp.status_code,
        "error": None if resp.is_success else body,
        "hf": model,
        "usage": body.get("usage") if isinstance(body, dict) else None,
    }


async def async_chat(
    client: httpx.AsyncClient,
    alias: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int | None = 42,
) -> dict[str, Any]:
    if alias not in BACKENDS:
        return {
            "alias": alias,
            "ok": False,
            "content": "",
            "latency_ms": 0.0,
            "gpu_seconds": 0.0,
            "http_status": 0,
            "error": f"unknown alias {alias}",
        }
    base, model = BACKENDS[alias]
    started = time.perf_counter()
    try:
        resp = await client.post(
            f"{base}/chat/completions",
            json=_payload(model, prompt, temperature, max_tokens, seed),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "alias": alias,
            "ok": False,
            "content": "",
            "latency_ms": round(elapsed, 1),
            "gpu_seconds": round(elapsed / 1000.0, 4),
            "http_status": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "hf": model,
        }
    return _result(alias, model, resp, (time.perf_counter() - started) * 1000)


def sync_chat(
    client: httpx.Client,
    alias: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int | None = 42,
) -> dict[str, Any]:
    if alias not in BACKENDS:
        return {
            "alias": alias,
            "ok": False,
            "content": "",
            "latency_ms": 0.0,
            "gpu_seconds": 0.0,
            "http_status": 0,
            "error": f"unknown alias {alias}",
        }
    base, model = BACKENDS[alias]
    started = time.perf_counter()
    try:
        resp = client.post(
            f"{base}/chat/completions",
            json=_payload(model, prompt, temperature, max_tokens, seed),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "alias": alias,
            "ok": False,
            "content": "",
            "latency_ms": round(elapsed, 1),
            "gpu_seconds": round(elapsed / 1000.0, 4),
            "http_status": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "hf": model,
        }
    return _result(alias, model, resp, (time.perf_counter() - started) * 1000)
