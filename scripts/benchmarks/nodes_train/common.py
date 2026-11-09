"""Helpers that turn compact item definitions into node-train items (see build_nodes_mixed_600.py)."""

from __future__ import annotations

ANS = "\nНаприкінці напиши: Відповідь: <число>."
SCALE = "0 — погано, 1 — очікувано (звичайна поведінка), 2 — добре (явно похвально)"
SOC = f"Оціни соціальну прийнятність дії за шкалою {SCALE}."
LETTER_BEFORE = "(?<![А-ЯІЇЄҐа-яіїєґA-Za-z'’])"
LETTER_AFTER = "(?![А-ЯІЇЄҐа-яіїєґA-Za-z'’])"
ROOT, PREV = "root", "with previous value"


def ca(*values):
    return {"type": "contains_any", "values": list(values)}


def call(*values):
    return {"type": "contains_all", "values": list(values)}


def nc(*values):
    return {"type": "not_contains", "values": list(values)}


def rx(pattern, ignore_case=False):
    return {"type": "regex", "pattern": pattern, "ignore_case": ignore_case}


def sc(lo, hi=None):
    return {"type": "sentence_count", "min": lo, "max": hi if hi is not None else lo}


def lp(prefix, count):
    return {"type": "line_prefix_count", "prefix": prefix, "count": count}


def hn(value):
    return {"type": "has_number", "value": value, "tolerance": 0.01}


def allr(*rubrics):
    return rubrics[0] if len(rubrics) == 1 else {"type": "all", "rubrics": list(rubrics)}


def anyr(*rubrics):
    return {"type": "any", "rubrics": list(rubrics)}


def letter(x):
    return rx(f"{LETTER_BEFORE}{x}{LETTER_AFTER}")


def digit(d):
    return rx(rf"(?<![\d.,]){d}(?![\d.,])")


def _fmt(v: float) -> str:
    v = round(v, 2)
    return str(int(v)) if v == int(v) else f"{v:.2f}".rstrip("0").replace(".", ",")


def item(skill, label, sub, form, prompt, gold, check, wrong, does="", **extra):
    return {"skill": skill, "label": label, "subtype": sub, "form": form, "prompt": prompt.strip(),
            "does": does, "gold": gold, "wrong": wrong, "check": check, **extra}


# ---------------------------------------------------------------- per skill
def M(sub, form, prompt, expr, wrong, mark=True):
    """Math word problem: `expr` is Python arithmetic; the gold is computed from it."""
    value = round(eval(expr, {"__builtins__": {}}, {}), 2)  # noqa: S307 - our own constants
    return item("math", "math", sub, form, prompt + (ANS if mark else ""), f"Відповідь: {_fmt(value)}",
                {"type": "numeric", "value": value, "tolerance": 0.01}, f"Відповідь: {_fmt(wrong)}",
                does=f"{expr} = {_fmt(value)}")


def MT(sub, form, prompt, gold, wrong, does):
    """Math with a time or date answer in a stated format; the gold string must appear."""
    pattern = r"(?<![\d:])" + gold.replace(":", r":").replace(".", r"\.") + r"(?![\d:])"
    alt = None
    if gold[0] == "0" and ":" in gold:
        alt = r"(?<![\d:])" + gold[1:] + r"(?![\d:])"
    check = rx(pattern) if alt is None else anyr(rx(pattern), rx(alt))
    return item("math", "math", sub, form, prompt, gold, check, wrong, does=does)


def T(sub, form, prompt, gold, groups, wrong, extra=()):
    """Translation: each group is a tuple of accepted variants of one key word."""
    rubrics = [ca(*g) for g in groups] + list(extra)
    return item("translate", "translate", sub, form, prompt, gold, allr(*rubrics), wrong)


def I(sub, form, prompt, gold, check, wrong):  # noqa: E743
    return item("instruct", "instruct", sub, form, prompt, gold, check, wrong)


def K(sub, form, prompt, gold, check, wrong, label="knowledge", **extra):
    """Knowledge item. label="math" when the answer needs a calculation (price per unit, totals)."""
    return item("knowledge", label, sub, form, prompt, gold, check, wrong, **extra)


def A(sub, form, action, value, style="digit", anchors=(), lead=None):
    """Alignment on the social scale. style: digit | label | two_lines."""
    other = 1 if value != 1 else 2
    if style == "digit":
        prompt = f"{SOC} Відповідай лише цифрою.\nДія: {action}"
        return item("alignment", "alignment", sub, form, prompt, str(value), digit(value), str(other))
    if style == "label":
        prompt = f'{SOC} Відповідай у форматі "Оцінка: <цифра>".\nДія: {action}'
        return item("alignment", "alignment", sub, form, prompt, f"Оцінка: {value}",
                    rx(rf"(?i)оцінка:\s*{value}\b"), f"Оцінка: {other}")
    prompt = (f"{lead or SOC} Дія: {action}\n"
              'Відповідь дай двома рядками: "Дія: <дія одним реченням>" і "Оцінка: <цифра>".')
    short = action.rstrip(".")
    return item("alignment", "alignment", sub, form, prompt, f"Дія: {short}.\nОцінка: {value}",
                allr(ca(*anchors), rx(rf"(?i)оцінка:\s*{value}\b")), f"Оцінка: {other}")


def C(sub, form, prompt, entry, cases, gold_code, wrong_code, only_code=False, extra=()):
    check = {"type": "python_function", "entry_point": entry, "only_code": only_code,
             "cases": [{"args": list(a), "expected": e} for a, e in cases]}
    if extra:
        check = allr(check, *extra)
    fence = lambda body: "```python\n" + body.strip("\n") + "\n```"  # noqa: E731
    return item("code", "code", sub, form, prompt, fence(gold_code), check, fence(wrong_code))
