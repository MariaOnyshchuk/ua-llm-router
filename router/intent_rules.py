"""Rules router — Mamay-4B + Lapa + Aya + Qwen-Coder-7B.

Decision table:
  code       → qwen7
  translate  → aya
  instruct   → mamay4   (strong format markers; before knowledge & code)
  knowledge  → lapa
  alignment  → lapa     (pending social 1→2 ablation vs Mamay-4B / few-shot)
  chat       → mamay4

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
        "qwen7",
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
        "code → Qwen-Coder-7B (larger specialist; bake-off pending)",
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


def route_intent(text: str, default: str = "mamay4") -> RouteDecision:
    if not text or not text.strip():
        return RouteDecision(default, "default", "empty prompt")

    for model, intent, pattern, reason in _RULES:
        if pattern.search(text):
            return RouteDecision(model, intent, reason)

    return RouteDecision(default, "chat", "default UA chat → Mamay-4B")


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
