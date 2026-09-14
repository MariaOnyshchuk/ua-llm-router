"""Rules router — Mamay-4B + Lapa + Aya (+ Qwen-7B optional).

Decision table (week 7, v4-best specialists in the *rules*):
  code       → mamay4   (was qwen7; v4 code 0.984 vs 0.969)
  translate  → aya
  instruct   → mamay4
  knowledge  → lapa
  alignment  → lapa
  chat       → aya      (default; was mamay4; v4 chat 1.000 vs 0.984)

Backends:
  mamay4 → :8003
  lapa   → :8001
  aya    → :8005
  qwen7  → :8004
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    model: str
    intent: str
    reason: str


# (model, intent_bucket, pattern, reason) — first match wins.
#
# Instruct uses *strong* format markers only (not bare «тільки»): bare
# «тільки» appears inside ZNO passages and «Тільки код» code prompts and
# previously stole those buckets. Instruct is still above knowledge so
# «Що таке …» / years inside format tasks do not leak to Lapa, and above
# code so «Поверни ТІЛЬКИ … python …» stays instruct.
_RULES: list[tuple[str, str, re.Pattern[str], str]] = [
    (
        "aya",
        "translate",
        re.compile(
            r"("
            r"\b(translat\w*|переклад\w*|переклад(и|іть)?)\b|"
            r"\b(en\s*→\s*uk|uk\s*→\s*en|англійськ|українськ).{0,30}(переклад|translate)|"
            r"translate\s+(this|the|from|into|to)\b|"
            r"Переклади з (англійської|української)"
            r")",
            re.I | re.U,
        ),
        "translate → Aya Expanse 8B (cross-lineage multilingual; bake-off pending)",
    ),
    (
        "lapa",
        "alignment",
        re.compile(
            r"("
            r"\b(моральн\w*|етичн\w*|ethical|moral)\b|"
            r"\b(прийнятн\w*|неприйнятн\w*|acceptable|unacceptable)\b|"
            r"\b(погано|добре|очікувано|bad|good|expected)\b.{0,40}\b(0|1|2)\b|"
            r"відповідай лише цифрою|"
            r"соціальн\w* прийнятн"
            r")",
            re.I | re.U,
        ),
        "alignment → Lapa (pending social 1→2 ablation vs Mamay-4B)",
    ),
    (
        "mamay4",
        "instruct",
        re.compile(
            r"("
            r"\b(step[- ]by[- ]step|покроково|по пунктах)\b|"
            r"\b(follow|дотримуйся|виконай).{0,40}(instruction|інструкц|правил)|"
            r"\b(format|оформи|структуруй).{0,30}(json|yaml|markdown|таблиц)|"
            r"\b(system prompt|роль:|ти —|ти є)\b|"
            r"\b(must|обов['’]?язково|не менше ніж)\b|"
            r"\b(exactly)\b|"
            r"\bрівно\b|"
            r"Поверни ТІЛЬКИ|Відповідай ЛИШЕ|Відповідай ТІЛЬКИ|"
            r"ВЕЛИКИМИ ЛІТЕРАМИ|"
            r"починається зі слова|"
            r"одним словом|"
            r"Нічого поза блоком"
            r")",
            re.I | re.U,
        ),
        "instruct → Mamay-4B (strong format constraints before knowledge/code)",
    ),
    (
        "mamay4",
        "code",
        re.compile(
            r"("
            r"\b(code|coding|program|debug|refactor|leetcode|github|sql|regex)\b|"
            r"\b(python|javascript|typescript|rust|golang|java|cpp|c\+\+)\b|"
            r"(напиши|виправ|зрефактор).{0,40}(код|скрипт|функц)|"
            r"\b(функція|класс|клас)\b|"
            r"```|"
            r"Тільки код"
            r")",
            re.I | re.U,
        ),
        "code → Mamay-4B (v4 winner vs Qwen-Coder-7B)",
    ),
    (
        "lapa",
        "knowledge",
        re.compile(
            r"("
            # factoid openers (UA hand-written know-*)
            r"^(У якому|В якому|Яка|Яке|Який|Які|Скільки|Хто автор|Хто написав|"
            r"Хто така|Хто такий|Назви|Назвіть)\b|"
            r"\b(who is|what is|when did|хто так(ий|а)|що так(е|ий)|коли)\b|"
            r"\b(новин\w*|актуальн\w*|станом на|as of|latest|recent)\b|"
            r"\b(20(2[4-9]|3[0-9]))\b|"
            r"\b(wikipedia|вікіпеді)\b|"
            r"тестове завдання ЗНО|Варіанти:|"
            r"\b(ЗНО|НМТ)\b"
            r")",
            re.I | re.U | re.M,
        ),
        "knowledge → Lapa (bake-off +0.16 vs Mamay-4B)",
    ),
]


# Allowed router targets (must match LiteLLM aliases / cluster BACKENDS)
# Multi-lineage pool: Mamay-4B/Lapa (Gemma-UA), Aya (Cohere), Qwen7 (Qwen).
ROUTER_MODELS = frozenset({"mamay4", "lapa", "aya", "qwen7"})

# Gold-bucket “best specialist” map from mixed_ua_v4 solos (week 6).
# This is an oracle given suite labels — not the production rules router.
# Alignment stays Lapa: Mamay-4B few-shot 0.781 was a 32-item slice, not full-suite.
BEST_BY_BUCKET: dict[str, str] = {
    "chat": "aya",  # 1.000 vs Mamay-4B 0.984
    "instruct": "mamay4",  # tie 0.781 with Aya; keep UA IF specialist
    "knowledge": "lapa",  # 0.750
    "alignment": "lapa",  # 0.750 with few-shot; same as matrix
    "code": "mamay4",  # 0.984 vs Qwen-7B 0.969 — Qwen drops out of this map
    "translate": "aya",  # 0.821
}


# Live / eval profiles. v2 is the product table baked into _RULES.
# v1 remaps after the match: code → Qwen-7B, unmatched chat → Mamay-4B.
ROUTER_PROFILES: dict[str, dict[str, object]] = {
    "v1": {
        "label": "Rules v1",
        "remap": {"code": "qwen7"},
        "default": "mamay4",
        "blurb": "code → Qwen-7B · chat → Mamay-4B",
    },
    "v2": {
        "label": "Rules v2",
        "remap": {},
        "default": "aya",
        "blurb": "code → Mamay-4B · chat → Aya",
    },
    "knn": {
        "label": "Learned kNN",
        "remap": {},
        "default": "aya",
        "blurb": "multilingual embedding kNN (Hybrid-LLM-style)",
        "learned": "knn",
    },
    "clf": {
        "label": "Learned classifier",
        "remap": {},
        "default": "aya",
        "blurb": "multilingual embedding logistic regression",
        "learned": "clf",
    },
}


def route_intent(
    text: str,
    default: str | None = None,
    *,
    profile: str = "v2",
    pool: frozenset[str] | set[str] | None = None,
    intent_hint: str | None = None,
) -> RouteDecision:
    spec = ROUTER_PROFILES.get(profile) or ROUTER_PROFILES["v2"]
    fallback = default if default is not None else str(spec["default"])
    remap = spec["remap"] if isinstance(spec["remap"], dict) else {}
    hinted = (intent_hint or "").strip().lower()
    learned_kind = spec.get("learned") if isinstance(spec.get("learned"), str) else None
    if learned_kind and not hinted:
        from router.learned import route_learned

        return route_learned(text, kind=learned_kind, default=fallback, pool=pool)

    hinted_rule = next(
        ((model, reason) for model, intent, _pattern, reason in _RULES if intent == hinted),
        None,
    )
    if hinted == "chat":
        decision = RouteDecision(fallback, "chat", f"planner intent hint → {fallback}")
    elif hinted_rule is not None:
        model, reason = hinted_rule
        target = str(remap.get(hinted, model))
        decision = RouteDecision(target, hinted, f"planner intent hint; {reason}")
    elif not text or not text.strip():
        decision = RouteDecision(fallback, "default", "empty prompt")
    else:
        decision = None
        for model, intent, pattern, reason in _RULES:
            if pattern.search(text):
                target = str(remap.get(intent, model))
                if target != model:
                    reason = f"{intent} → {target} (profile {profile})"
                decision = RouteDecision(target, intent, reason)
                break
        if decision is None:
            chat_reason = (
                "default UA chat → Mamay-4B"
                if fallback == "mamay4"
                else "default UA chat → Aya"
            )
            decision = RouteDecision(fallback, "chat", chat_reason)

    if pool:
        allowed = {m for m in pool if m in ROUTER_MODELS}
        if allowed and decision.model not in allowed:
            pick = next(
                (m for m in ("aya", "mamay4", "lapa", "qwen7") if m in allowed),
                decision.model,
            )
            decision = RouteDecision(
                pick,
                decision.intent,
                f"{decision.reason}; {decision.model} not in pool → {pick}",
            )
    return decision


def extract_user_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages or []:
        role = m.get("role")
        if role in ("user", "system"):
            content = m.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
    return "\n".join(parts)
