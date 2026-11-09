"""Intent classification by a small model: one short call that returns one word.

The rules in `intent_rules.py` match wording. Requests in everyday language often
carry none of the marker words ("Як сказати англійською «дякую»?"), so they fall
through to chat. This module asks a specialist for the intent instead. It is not
used by any existing profile; `scripts/analysis/intent_natural_eval.py` measures it.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from router.backends import BACKENDS

INTENTS = ("translate", "knowledge", "alignment", "instruct", "code", "math", "chat")
# Same table as rules v2: the classifier changes the intent, not the model map.
INTENT_MODEL = {"translate": "aya", "chat": "aya", "knowledge": "lapa", "alignment": "lapa",
                "instruct": "mamay4", "code": "mamay4", "math": "lapa"}


PROMPT = """Визнач тип запиту користувача. Відповідай ОДНИМ словом зі списку.
Якщо підходить кілька типів, обери той, що стоїть вище в списку.

code — про програмування, код, запити до бази, команди терміналу (навіть якщо є вимоги до довжини чи формату);
translate — просять перекласти текст або питають, як щось сказати іншою мовою;
math — треба щось порахувати: гроші, відсотки, час, дати, одиниці виміру (навіть якщо є вимога до формату відповіді);
instruct — просять написати текст і ставлять вимоги до форми: кількість речень чи слів, формат, які слова вживати;
alignment — просять оцінити вчинок: чи це правильно, чесно, ввічливо, припустимо;
knowledge — питання про факт, пояснення, як щось влаштовано;
chat — звичайна розмова, порада, привітання, а також усе, що не підходить під інші типи.

Запит не виконуй, лише назви тип.

Запит:
{request}

Тип:"""

_WORD = re.compile(r"[a-z]+")


def parse(reply: str) -> str | None:
    """First allowed intent word in the reply, if any."""
    for word in _WORD.findall((reply or "").lower()):
        if word in INTENTS:
            return word
    return None


def classify(client: httpx.Client, request: str, alias: str = "mamay4") -> dict[str, Any]:
    """Ask `alias` for the intent. Returns intent (None if unusable), raw reply and latency."""
    base, model = BACKENDS[alias][0], BACKENDS[alias][1]
    resp = client.post(f"{base}/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": PROMPT.format(request=request)}],
        "temperature": 0, "seed": 42, "max_tokens": 6})
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"] or ""
    return {"intent": parse(reply), "reply": reply.strip(), "latency_ms": resp.elapsed.total_seconds() * 1000}
