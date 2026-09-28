"""Constrained workflow templates for composite requests.

Maps a user prompt to one of a small set of permitted multi-step templates.
Ambiguous or unsupported requests fall back to a single-hop plan rather than
inventing a free-form workflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from router.intent_rules import route_intent
from router.orchestrator import Plan, PlanStep, ensure_dependency_placeholders, validate_plan

Renderer = Callable[[str, dict[str, str]], str]


@dataclass(frozen=True)
class WorkflowTemplate:
    id: str
    intents: tuple[str, ...]
    # (step_id, intent, depends_on, prompt_renderer_name)
    step_specs: tuple[tuple[str, str, tuple[str, ...], str], ...]
    matchers: tuple[re.Pattern[str], ...]
    # Prefer this template when several matchers fire (higher wins).
    priority: int = 0


def _extract_spec_block(prompt: str) -> str:
    match = re.search(r"(?is)Specification:\s*(.+)$", prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _extract_source_block(prompt: str) -> str:
    match = re.search(r"(?is)Source:\s*(.+)$", prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _extract_notice_block(prompt: str) -> str:
    match = re.search(r"(?is)Оголошення:\s*(.+)$", prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _extract_fn_name(prompt: str) -> str:
    match = re.search(r"`([A-Za-z_][\w]*)`", prompt)
    return match.group(1) if match else "solution"


def _qa_body(prompt: str) -> str:
    """Drop the instruction preamble; keep question + options."""
    parts = prompt.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else prompt.strip()


RENDERERS: dict[str, Renderer] = {
    "tc_translate": lambda user, ctx: (
        f"Переклади українською цю специфікацію без додавання вимог:\n{ctx['spec']}"
    ),
    "tc_implement": lambda user, ctx: (
        f"Реалізуй Python-функцію `{ctx['fn']}` за специфікацією нижче. "
        "Поверни лише блок ```python.\n\n{{translate.content}}"
    ),
    "ke_answer": lambda user, ctx: (
        f"{ctx['body']}\nВідповідай лише літерою А, Б або В."
    ),
    "ke_explain": lambda user, ctx: (
        "Дай фінальну відповідь рівно у двох рядках:\n"
        "Відповідь: <літера>\nПояснення: <одне речення>\n"
        f"{ctx['body']}\n"
        "Попередня відповідь: {{answer.content}}"
    ),
    "tkw_translate": lambda user, ctx: (
        f"Точно переклади джерело українською:\n{ctx['source']}"
    ),
    "tkw_extract": lambda user, ctx: (
        "Витягни з тексту subject, number і place. Поверни лише JSON.\n"
        "{{translate.content}}"
    ),
    "tkw_write": lambda user, ctx: (
        "Нормалізуй дані у валідний JSON. Рівно три ключі: subject, number, "
        "place. Без markdown і пояснень.\n{{extract.content}}"
    ),
    "ecw_extract": lambda user, ctx: (
        "Витягни з оголошення короткий subject і number (телефон/дату/номер). "
        "Поверни лише JSON з ключами subject і number.\n"
        f"{ctx['notice']}"
    ),
    "ecw_classify": lambda user, ctx: (
        "Класифікуй оголошення. topic ∈ {utilities, culture, education}; "
        "urgency ∈ {low, medium, high}. Поверни JSON з topic і urgency.\n"
        f"Оголошення:\n{ctx['notice']}\n"
        "Витяг: {{extract.content}}"
    ),
    "ecw_write": lambda user, ctx: (
        "Збери фінальний JSON рівно з ключами topic, urgency, subject, number. "
        "Без markdown і пояснень.\n"
        "Витяг: {{extract.content}}\n"
        "Класифікація: {{classify.content}}"
    ),
    "tsf_translate": lambda user, ctx: (
        f"Точно переклади повідомлення українською:\n{ctx['source']}"
    ),
    "tsf_summarize": lambda user, ctx: (
        "Стисни український текст до одного речення-резюме без нових фактів.\n"
        "{{translate.content}}"
    ),
    "tsf_format": lambda user, ctx: (
        "Поверни лише валідний JSON з ключами topic, when, action. "
        "Значення topic/when/action бери з резюме (можна латиницею, як у джерелі). "
        "Без markdown і пояснень.\n{{summarize.content}}"
    ),
}


TEMPLATES: tuple[WorkflowTemplate, ...] = (
    WorkflowTemplate(
        id="translate_code",
        intents=("translate", "code"),
        step_specs=(
            ("translate", "translate", (), "tc_translate"),
            ("implement", "code", ("translate",), "tc_implement"),
        ),
        matchers=(
            re.compile(
                r"(?is)переклади.{0,80}специфікац.{0,120}python|"
                r"Specification:|"
                r"лише блок\s*```python"
            ),
        ),
        priority=30,
    ),
    WorkflowTemplate(
        id="knowledge_explain",
        intents=("knowledge", "instruct"),
        step_specs=(
            ("answer", "knowledge", (), "ke_answer"),
            ("explain", "instruct", ("answer",), "ke_explain"),
        ),
        matchers=(
            re.compile(
                r"(?is)рівно два рядки.{0,40}Відповідь:|"
                r"Пояснення:\s*<одне речення>"
            ),
        ),
        priority=20,
    ),
    WorkflowTemplate(
        id="translate_knowledge_write",
        intents=("translate", "knowledge", "instruct"),
        step_specs=(
            ("translate", "translate", (), "tkw_translate"),
            ("extract", "knowledge", ("translate",), "tkw_extract"),
            ("write", "instruct", ("extract",), "tkw_write"),
        ),
        matchers=(
            re.compile(
                r"(?is)subject.{0,20}number.{0,20}place|"
                r"переклади його українською,\s*витягни три факти"
            ),
        ),
        priority=40,
    ),
    WorkflowTemplate(
        id="extract_classify_write",
        intents=("knowledge", "knowledge", "instruct"),
        step_specs=(
            ("extract", "knowledge", (), "ecw_extract"),
            ("classify", "knowledge", ("extract",), "ecw_classify"),
            ("write", "instruct", ("extract", "classify"), "ecw_write"),
        ),
        matchers=(
            re.compile(
                r"(?is)topic.{0,30}urgency.{0,30}subject.{0,30}number|"
                r"utilities,\s*culture,\s*education"
            ),
        ),
        priority=50,
    ),
    WorkflowTemplate(
        id="translate_summarize_format",
        intents=("translate", "instruct", "instruct"),
        step_specs=(
            ("translate", "translate", (), "tsf_translate"),
            ("summarize", "instruct", ("translate",), "tsf_summarize"),
            ("format", "instruct", ("summarize",), "tsf_format"),
        ),
        matchers=(
            re.compile(
                r"(?is)topic.{0,20}when.{0,20}action|"
                r"одне речення-резюме"
            ),
        ),
        priority=45,
    ),
)

TEMPLATE_BY_ID = {t.id: t for t in TEMPLATES}


def build_context(user_prompt: str) -> dict[str, str]:
    return {
        "spec": _extract_spec_block(user_prompt),
        "source": _extract_source_block(user_prompt),
        "notice": _extract_notice_block(user_prompt),
        "fn": _extract_fn_name(user_prompt),
        "body": _qa_body(user_prompt),
    }


def select_template(user_prompt: str) -> WorkflowTemplate | None:
    hits: list[WorkflowTemplate] = []
    for template in TEMPLATES:
        if any(m.search(user_prompt or "") for m in template.matchers):
            hits.append(template)
    if not hits:
        return None
    hits.sort(key=lambda t: t.priority, reverse=True)
    return hits[0]


def render_template_plan(template: WorkflowTemplate, user_prompt: str) -> Plan:
    ctx = build_context(user_prompt)
    steps: list[PlanStep] = []
    for step_id, intent, depends_on, renderer_name in template.step_specs:
        renderer = RENDERERS[renderer_name]
        steps.append(
            PlanStep(
                id=step_id,
                intent=intent,
                depends_on=depends_on,
                prompt=renderer(user_prompt, ctx),
            )
        )
    plan = Plan(steps=tuple(steps), source=f"template:{template.id}")
    plan = ensure_dependency_placeholders(plan)
    validate_plan(plan)
    return plan


def fallback_single_hop(user_prompt: str) -> Plan:
    decision = route_intent(user_prompt)
    return Plan(
        steps=(
            PlanStep(
                id="fallback",
                intent=decision.intent,
                depends_on=(),
                prompt=user_prompt,
            ),
        ),
        source="single_hop_fallback",
    )


def select_plan(user_prompt: str) -> tuple[Plan, dict[str, Any]]:
    """Select a constrained template plan or fall back to one hop."""
    template = select_template(user_prompt)
    if template is None:
        plan = fallback_single_hop(user_prompt)
        return plan, {
            "mode": "template",
            "valid": True,
            "template_id": None,
            "fallback": True,
            "fallback_reason": "no_template_match",
            "repaired": False,
            "attempts": [],
        }
    try:
        plan = render_template_plan(template, user_prompt)
    except (ValueError, KeyError) as exc:
        plan = fallback_single_hop(user_prompt)
        return plan, {
            "mode": "template",
            "valid": False,
            "template_id": template.id,
            "fallback": True,
            "fallback_reason": f"render_error:{exc}",
            "repaired": False,
            "attempts": [],
        }
    return plan, {
        "mode": "template",
        "valid": True,
        "template_id": template.id,
        "fallback": False,
        "fallback_reason": None,
        "repaired": False,
        "attempts": [],
    }
