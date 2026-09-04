"""Typed plans and bounded dependency execution for composite requests."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

from router.intent_rules import ROUTER_MODELS, route_intent

ALLOWED_INTENTS = frozenset({"translate", "knowledge", "instruct", "code", "alignment", "chat"})
PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z][\w-]*)\.content\}\}")
AsyncCaller = Callable[[str, str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class PlanStep:
    id: str
    intent: str
    prompt: str
    depends_on: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanStep":
        return cls(
            id=str(raw.get("id") or "").strip(),
            intent=str(raw.get("intent") or "").strip().lower(),
            prompt=str(raw.get("prompt") or "").strip(),
            depends_on=tuple(str(x) for x in (raw.get("depends_on") or [])),
        )


@dataclass(frozen=True)
class Plan:
    steps: tuple[PlanStep, ...]
    source: str = "unknown"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, source: str = "unknown") -> "Plan":
        return cls(
            steps=tuple(PlanStep.from_dict(x) for x in (raw.get("steps") or [])),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "steps": [asdict(step) for step in self.steps]}


def validate_plan(plan: Plan, *, max_steps: int = 4) -> None:
    if not 1 <= len(plan.steps) <= max_steps:
        raise ValueError(f"plan must contain 1-{max_steps} steps")
    seen: set[str] = set()
    for step in plan.steps:
        if not re.fullmatch(r"[a-zA-Z][\w-]{0,39}", step.id):
            raise ValueError(f"invalid step id: {step.id!r}")
        if step.id in seen:
            raise ValueError(f"duplicate step id: {step.id}")
        if step.intent not in ALLOWED_INTENTS:
            raise ValueError(f"invalid intent for {step.id}: {step.intent}")
        if not step.prompt or len(step.prompt) > 6000:
            raise ValueError(f"invalid prompt for {step.id}")
        if any(dep not in seen for dep in step.depends_on):
            raise ValueError(f"{step.id}: dependencies must refer to earlier steps")
        referenced = set(PLACEHOLDER_RE.findall(step.prompt))
        if not referenced.issubset(set(step.depends_on)):
            missing = sorted(referenced - set(step.depends_on))
            raise ValueError(f"{step.id}: placeholders missing from depends_on: {missing}")
        seen.add(step.id)


def render_prompt(step: PlanStep, results: dict[str, dict[str, Any]]) -> str:
    def replace(match: re.Match[str]) -> str:
        dep = match.group(1)
        if dep not in results:
            raise ValueError(f"{step.id}: dependency {dep} has no result")
        return str(results[dep].get("content") or "")

    return PLACEHOLDER_RE.sub(replace, step.prompt)


def plan_signature(plan: Plan) -> list[dict[str, Any]]:
    """Stable structure for workflow matching; ignores arbitrary generated IDs."""
    index = {step.id: i for i, step in enumerate(plan.steps)}
    return [
        {
            "intent": step.intent,
            "depends_on": sorted(index[dep] for dep in step.depends_on),
        }
        for step in plan.steps
    ]


async def execute_plan(
    plan: Plan,
    caller: AsyncCaller,
    *,
    profile: str = "v2",
    pool: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    validate_plan(plan)
    results: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    total_latency = 0.0
    total_gpu_seconds = 0.0

    for step in plan.steps:
        prompt = render_prompt(step, results)
        decision = route_intent(
            prompt,
            profile=profile,
            pool=pool,
            intent_hint=step.intent,
        )
        alias = decision.model
        if alias not in ROUTER_MODELS:
            raise ValueError(f"router selected unsupported model: {alias}")
        hop = await caller(alias, prompt)
        result = {
            **hop,
            "id": step.id,
            "intent": step.intent,
            "depends_on": list(step.depends_on),
            "prompt": prompt,
            "route": {
                "model": alias,
                "detected_intent": decision.intent,
                "declared_intent": step.intent,
                "reason": decision.reason,
            },
        }
        results[step.id] = result
        trace.append(result)
        total_latency += float(hop.get("latency_ms") or 0)
        total_gpu_seconds += float(hop.get("gpu_seconds") or 0)
        if not hop.get("ok"):
            break

    complete = len(trace) == len(plan.steps) and all(step.get("ok") for step in trace)
    final = trace[-1] if trace else {}
    return {
        "plan": plan.to_dict(),
        "steps": trace,
        "complete": complete,
        "final_output": final.get("content") or "",
        "final_step": final.get("id"),
        "calls": len(trace),
        "latency_ms": round(total_latency, 1),
        "gpu_seconds": round(total_gpu_seconds, 4),
    }
