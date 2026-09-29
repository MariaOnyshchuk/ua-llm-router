"""Debug UI backend: streaming chat, routing trace, composite view, compare, benchmark browser.

Mounted by router.app at /debug. Local use only; it reaches the specialists through the
same BACKENDS table as the eval runners (SSH tunnels to the lab, or scripts/mock_backends.py).

Every run is appended to logs/debug_sessions/<date>.jsonl. logs/ is git-ignored: benchmark
prompts and model answers must never reach git (AGENTS.md).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.backends import BACKENDS  # noqa: E402
from router.intent_rules import _RULES, ROUTER_MODELS, ROUTER_PROFILES, route_intent  # noqa: E402
from router.orchestrator import Plan, execute_plan, plan_signature  # noqa: E402
from router.planner import PLANNER_PROFILES, generate_plan  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOG_DIR = ROOT / "logs" / "debug_sessions"
DETAIL_GLOBS = ("results/v6_*/scores_detail.jsonl", "results/v6_*/scored/*detail.jsonl")

SUITES: dict[str, dict[str, Any]] = {
    "v6": {"path": "benchmarks/mixed_ua_v6.jsonl", "kind": "single", "note": "claim suite, 2414 items"},
    "v4": {"path": "benchmarks/mixed_ua_v4_balanced.jsonl", "kind": "single", "note": "dev suite, wiring checks"},
    "agentcoma": {
        "path": "benchmarks/agentcoma_uk_200_composite.jsonl",
        "merged": "benchmarks/agentcoma_uk_200_merged.jsonl",
        "kind": "composite",
        "note": "AgentCoMa-UK 200, local only",
    },
}

router = APIRouter(prefix="/debug")


# ----------------------------------------------------------------------------- helpers


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=8)
def _suite_items(suite: str) -> dict[str, dict[str, Any]]:
    spec = SUITES.get(suite)
    if spec is None:
        raise HTTPException(404, f"unknown suite {suite}")
    path = ROOT / spec["path"]
    if not path.is_file():
        raise HTTPException(404, f"{spec['path']} not found on this machine")
    items = {row["id"]: row for row in _read_jsonl(path)}
    if spec.get("merged") and (ROOT / spec["merged"]).is_file():
        for row in _read_jsonl(ROOT / spec["merged"]):
            if row["id"] in items:
                items[row["id"]]["_merged"] = row
    return items


@lru_cache(maxsize=16)
def _detail_scores(rel_path: str) -> dict[str, dict[str, dict[str, Any]]]:
    """system -> id -> row for one scores_detail file (path must stay under results/)."""
    path = (ROOT / rel_path).resolve()
    if not str(path).startswith(str(ROOT / "results")) or not path.is_file():
        raise HTTPException(400, "detail file must be under results/")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        if "score" in row:
            out.setdefault(row["system"], {})[row["id"]] = row
    return out


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _log(entry: dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
        path = LOG_DIR / f"{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break a run


def route_trace(text: str, profile: str) -> dict[str, Any]:
    """The decision plus every rule that matched, so alternatives are visible."""
    decision = route_intent(text, profile=profile)
    spec = ROUTER_PROFILES.get(profile) or ROUTER_PROFILES["v2"]
    learned = isinstance(spec.get("learned"), str)
    rules = []
    if not learned:
        for model, intent, pattern, reason in _RULES:
            m = pattern.search(text or "")
            rules.append(
                {
                    "intent": intent,
                    "model": model,
                    "reason": reason,
                    "matched": bool(m),
                    "match_text": (m.group(0)[:80] if m else None),
                }
            )
    return {
        "profile": profile,
        "model": decision.model,
        "intent": decision.intent,
        "reason": decision.reason,
        "learned": learned,
        "rules": rules,
        "default": spec.get("default"),
    }


def build_prompt(prompt: str, item: dict[str, Any] | None, align_prompt: str) -> str:
    """Same as the eval runner: routing sees the original text, the model sees the variant."""
    if not item or align_prompt in ("", "baseline"):
        return prompt
    from scripts.alignment_prompt_variants import apply_alignment_variant

    return apply_alignment_variant({**item, "prompt": prompt}, align_prompt)["prompt"]


class Params:
    def __init__(self, body: dict[str, Any]) -> None:
        self.temperature = float(body.get("temperature", 0.0))
        self.max_tokens = int(body.get("max_tokens") or 1024)
        seed = body.get("seed", 42)
        self.seed = None if seed in (None, "") else int(seed)
        self.profile = str(body.get("profile") or "v2")
        self.align_prompt = str(body.get("align_prompt") or "baseline")

    def dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "seed": self.seed}


async def stream_completion(
    client: httpx.AsyncClient, alias: str, prompt: str, params: Params
) -> AsyncIterator[dict[str, Any]]:
    """Yield request / delta / done events for one streamed backend call."""
    if alias not in BACKENDS:
        yield {"type": "error", "message": f"unknown alias {alias}"}
        return
    base, model = BACKENDS[alias]
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if params.seed is not None:
        body["seed"] = params.seed
    yield {"type": "request", "alias": alias, "url": f"{base}/chat/completions", "body": body}
    started = time.perf_counter()
    first: float | None = None
    text, finish, usage, chunks = [], None, None, 0
    try:
        async with client.stream("POST", f"{base}/chat/completions", json=body) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")[:800]
                yield {"type": "error", "message": f"HTTP {resp.status_code}: {detail}",
                       "http_status": resp.status_code}
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        if first is None:
                            first = time.perf_counter()
                        chunks += 1
                        text.append(delta)
                        yield {"type": "delta", "text": delta}
                    finish = choice.get("finish_reason") or finish
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return
    end = time.perf_counter()
    yield {
        "type": "done",
        "alias": alias,
        "model": model,
        "content": "".join(text),
        "finish_reason": finish,
        "usage": usage,
        "chunks": chunks,
        "ttft_ms": round((first - started) * 1000, 1) if first else None,
        "latency_ms": round((end - started) * 1000, 1),
        "http_status": 200,
    }


def score_single(item: dict[str, Any], content: str) -> dict[str, Any]:
    from scripts.score_results import score_row

    row = {**{k: v for k, v in item.items() if not k.startswith("_")}, "content": content, "http_status": 200}
    return score_row(row)


def score_composite(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    from scripts.score_composite_results import score_row as composite_row

    oracle = Plan.from_dict(item["oracle_plan"], source="oracle")
    plan_dict = result.get("plan")
    plan = Plan.from_dict(plan_dict, source="run") if plan_dict else None
    row = {
        "system": "debug",
        "id": item["id"],
        "family": item.get("family"),
        "oracle_plan": item["oracle_plan"],
        "final_step_expected": item["final_step"],
        "plan": plan_dict,
        "plan_valid": True if plan else None,
        "plan_match": bool(plan and plan_signature(plan) == plan_signature(oracle)),
        "steps": result.get("steps") or [],
        "content": result.get("final_output") or "",
        "complete": result.get("complete", False),
        "http_status": 200 if result.get("complete") else 0,
        "calls": result.get("calls", 0),
        "latency_ms": result.get("latency_ms", 0),
        "gpu_seconds": result.get("gpu_seconds", 0),
    }
    return composite_row(row)


async def _merge(sources: dict[str, AsyncIterator[dict[str, Any]]]) -> AsyncIterator[dict[str, Any]]:
    """Interleave several event streams, tagging each event with its lane."""
    queue: asyncio.Queue = asyncio.Queue()

    async def pump(side: str, gen: AsyncIterator[dict[str, Any]]) -> None:
        try:
            async for event in gen:
                await queue.put({**event, "side": side})
        finally:
            await queue.put({"type": "_end", "side": side})

    tasks = [asyncio.create_task(pump(side, gen)) for side, gen in sources.items()]
    remaining = len(tasks)
    try:
        while remaining:
            event = await queue.get()
            if event["type"] == "_end":
                remaining -= 1
            else:
                yield event
    finally:
        for task in tasks:
            task.cancel()


async def _lane(
    side: str,
    mode: str,
    model: str,
    prompt: str,
    params: Params,
    item: dict[str, Any] | None,
    suite: str,
    client: httpx.AsyncClient,
) -> AsyncIterator[dict[str, Any]]:
    """One result lane: 'router' routes on the original text, 'model' forces one backend."""
    if mode == "router":
        trace = route_trace(prompt, params.profile)
        alias = trace["model"] if trace["model"] in ROUTER_MODELS else "mamay4"
        yield {"type": "route", "trace": trace, "alias": alias}
    else:
        alias = model
        yield {"type": "route", "trace": {"model": alias, "intent": "explicit", "reason": "model chosen by you",
                                          "rules": [], "profile": None}, "alias": alias}
    sent = build_prompt(prompt, item, params.align_prompt)
    yield {"type": "prompt", "text": sent, "changed": sent != prompt}
    final: dict[str, Any] | None = None
    async for event in stream_completion(client, alias, sent, params):
        if event["type"] == "done":
            final = event
        yield event
    if final is not None and item is not None and SUITES[suite]["kind"] == "single":
        try:
            yield {"type": "score", "score": score_single(item, final["content"])}
        except Exception as exc:  # noqa: BLE001
            yield {"type": "score", "score": {"score": None, "detail": f"scorer error: {exc}"}}


def _stream_response(events: AsyncIterator[dict[str, Any]], log_kind: str, meta: dict[str, Any]) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        record: dict[str, Any] = {"kind": log_kind, **meta, "lanes": {}}
        try:
            async for event in events:
                yield _sse(event)
                lane = record["lanes"].setdefault(event.get("side", "a"), {})
                if event["type"] in ("route", "prompt", "done", "score", "error", "plan", "final"):
                    lane[event["type"]] = {k: v for k, v in event.items() if k not in ("type", "side")}
        finally:
            _log(record)

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _item_for(body: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    suite = str(body.get("suite") or "")
    item_id = str(body.get("item_id") or "")
    if suite and item_id:
        items = _suite_items(suite)
        if item_id not in items:
            raise HTTPException(404, f"{item_id} not in {suite}")
        return suite, items[item_id]
    return suite or "v6", None


# ----------------------------------------------------------------------------- endpoints


@router.get("")
def page() -> FileResponse:
    return FileResponse(STATIC_DIR / "debug.html")


@router.get("/api/health")
async def health() -> dict[str, Any]:
    out: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for alias, (base, model) in BACKENDS.items():
            try:
                r = await client.get(f"{base}/models")
                data = (r.json().get("data") or [{}]) if r.status_code == 200 else [{}]
                out[alias] = {"up": r.status_code == 200, "base": base, "model": model,
                              "mock": bool(data[0].get("mock"))}
            except Exception:  # noqa: BLE001
                out[alias] = {"up": False, "base": base, "model": model}
    return {"backends": out, "profiles": sorted(ROUTER_PROFILES), "planners": sorted(PLANNER_PROFILES)}


@router.post("/api/route")
def route_only(body: dict[str, Any]) -> dict[str, Any]:
    return route_trace(str(body.get("prompt") or ""), str(body.get("profile") or "v2"))


@router.post("/api/chat")
async def chat(body: dict[str, Any]) -> StreamingResponse:
    """Stream one lane. mode=router routes on the text; mode=model forces `model`."""
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")
    params = Params(body)
    suite, item = _item_for(body)
    mode = str(body.get("mode") or "router")

    async def events() -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async for e in _lane("a", mode, str(body.get("model") or "mamay12"), prompt, params, item, suite, client):
                yield e

    return _stream_response(events(), "chat", {"mode": mode, "model": body.get("model"), "suite": suite,
                                               "item_id": body.get("item_id"), "params": params.dict()})


@router.post("/api/compare")
async def compare(body: dict[str, Any]) -> StreamingResponse:
    """Lane a = router, lane b = one chosen model, same prompt, run concurrently."""
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")
    params = Params(body)
    suite, item = _item_for(body)
    model_b = str(body.get("model_b") or "mamay12")

    async def events() -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            lanes = {
                "a": _lane("a", "router", "", prompt, params, item, suite, client),
                "b": _lane("b", "model", model_b, prompt, params, item, suite, client),
            }
            async for e in _merge(lanes):
                yield e

    return _stream_response(events(), "compare", {"model_b": model_b, "suite": suite,
                                                  "item_id": body.get("item_id"), "params": params.dict()})


@router.post("/api/composite")
async def composite(body: dict[str, Any]) -> StreamingResponse:
    """Plan and run a composite request; emit plan, each specialist call, and the final result."""
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")
    params = Params(body)
    params.max_tokens = int(body.get("max_tokens") or 256)
    suite, item = _item_for(body)
    planner = str(body.get("planner_profile") or "hybrid")
    use_oracle = bool(body.get("use_oracle")) and item is not None and "oracle_plan" in item
    if planner not in PLANNER_PROFILES:
        raise HTTPException(400, f"planner_profile must be one of {sorted(PLANNER_PROFILES)}")

    async def events() -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue()

        async def run() -> None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async def caller(alias: str, step_prompt: str) -> dict[str, Any]:
                    await queue.put({"type": "call_start", "alias": alias, "prompt": step_prompt})
                    hop = None
                    started = time.perf_counter()
                    from router.client import async_chat

                    hop = await async_chat(client, alias, step_prompt, temperature=params.temperature,
                                           max_tokens=params.max_tokens, seed=params.seed)
                    await queue.put({"type": "call_done", "alias": alias, "ok": hop.get("ok"),
                                     "content": hop.get("content"), "latency_ms": hop.get("latency_ms"),
                                     "usage": hop.get("usage"), "finish_reason": hop.get("finish_reason"),
                                     "wall_ms": round((time.perf_counter() - started) * 1000, 1)})
                    return hop

                try:
                    if use_oracle:
                        plan = Plan.from_dict(item["oracle_plan"], source="oracle")
                        planning: dict[str, Any] = {"valid": True, "attempts": [], "mode": "oracle"}
                    else:
                        plan, planning = await generate_plan(prompt, caller, profile=planner)
                    await queue.put({"type": "plan", "plan": plan.to_dict(), "planning": planning})
                    execution = await execute_plan(plan, caller, profile=params.profile)
                    final: dict[str, Any] = {"type": "final", "execution": execution}
                    if item is not None and SUITES[suite]["kind"] == "composite":
                        final["score"] = score_composite(item, execution)
                    await queue.put(final)
                except Exception as exc:  # noqa: BLE001
                    await queue.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                finally:
                    await queue.put({"type": "_end"})

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event["type"] == "_end":
                    break
                yield event
        finally:
            task.cancel()

    return _stream_response(events(), "composite", {"planner": planner, "oracle": use_oracle, "suite": suite,
                                                    "item_id": body.get("item_id"), "params": params.dict()})


# ----------------------------------------------------------------------------- benchmark browser


@router.get("/api/bench/suites")
def bench_suites() -> list[dict[str, Any]]:
    out = []
    for name, spec in SUITES.items():
        out.append({"name": name, "kind": spec["kind"], "note": spec["note"],
                    "available": (ROOT / spec["path"]).is_file()})
    return out


@router.get("/api/bench/detail_files")
def detail_files() -> list[dict[str, Any]]:
    """Score files that hold item-level results, with the systems in each."""
    out = []
    for pattern in DETAIL_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            rel = str(path.relative_to(ROOT))
            try:
                systems = sorted(_detail_scores(rel))
            except Exception:  # noqa: BLE001
                continue
            out.append({"path": rel, "systems": systems})
    return out


@router.get("/api/bench/items")
def bench_items(
    suite: str,
    bucket: str = "",
    q: str = "",
    detail: str = "",
    system: str = "",
    max_score: float = 1.0,
    limit: int = 60,
    offset: int = 0,
) -> dict[str, Any]:
    """List items. With detail+system, only items that system scored below max_score (its failures)."""
    items = _suite_items(suite)
    scored = _detail_scores(detail).get(system, {}) if detail and system else None
    rows = []
    for item_id, item in items.items():
        if bucket and (item.get("bucket") or item.get("family")) != bucket:
            continue
        prompt = str(item.get("prompt") or "")
        if q and q.casefold() not in (prompt + item_id).casefold():
            continue
        score = None
        if scored is not None:
            if item_id not in scored or scored[item_id]["score"] >= max_score:
                continue
            score = scored[item_id]["score"]
        rows.append({"id": item_id, "bucket": item.get("bucket") or item.get("family"),
                     "snippet": prompt[:110].replace("\n", " "), "score": score,
                     "detail": scored[item_id].get("detail") if scored is not None else None})
    buckets = sorted({(i.get("bucket") or i.get("family") or "") for i in items.values()})
    return {"total": len(rows), "buckets": buckets, "items": rows[offset: offset + limit]}


@router.get("/api/bench/item")
def bench_item(suite: str, id: str) -> dict[str, Any]:
    items = _suite_items(suite)
    if id not in items:
        raise HTTPException(404, f"{id} not in {suite}")
    item = items[id]
    out = {k: v for k, v in item.items() if not k.startswith("_")}
    merged = item.get("_merged")
    if merged:
        out["steps_alone"] = {
            "commonsense": merged.get("question_commonsense_uk"),
            "commonsense_gold": merged.get("answers_commonsense_uk"),
            "math": merged.get("question_math_uk"),
            "math_gold": merged.get("answer_math"),
        }
    out["kind"] = SUITES[suite]["kind"]
    return out


@router.post("/api/score")
def score_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    suite, item = _item_for(body)
    if item is None:
        raise HTTPException(400, "suite and item_id required")
    if SUITES[suite]["kind"] != "single":
        raise HTTPException(400, "use the composite run for composite items")
    return score_single(item, str(body.get("content") or ""))


@router.get("/api/log")
def read_log(limit: int = 40) -> list[dict[str, Any]]:
    if not LOG_DIR.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(LOG_DIR.glob("*.jsonl"), reverse=True):
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] + rows
        if len(rows) >= limit:
            break
    return rows[-limit:][::-1]
