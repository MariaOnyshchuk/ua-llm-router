"""Enforce a few explicit surface constraints of a request on the answer, without a model.

On mixed_ua_v7 dev the whole gap between the rules router and Mamay-12B is in the
instruct bucket, and inside it in four constraints the specialists ignore: "no commas",
"wrap the whole answer in double quotes", "all lowercase", "all capital letters". Each of
them can be checked and repaired with plain string operations.

`detect` reads the constraints from the request text only (never from benchmark metadata).
`apply` returns the repaired answer and the list of repairs it made. Off by default; the
existing router profiles do not call it.
"""

from __future__ import annotations

import re

_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")

_NO_COMMA = re.compile(
    r"(?i)(\bне\b|\bбез\b|жодн\w*|утрима\w+|уника\w+|заборон\w+)[^.!?\n]{0,60}\bком(и|а|у|ою)?\b"
    r"|\bком(и|а)?\b[^.!?\n]{0,40}\bне\s+(дозвол|можна|повинн|має|мають)"
    r"|\b(no|without(\s+any)?|avoid(\s+using)?)\s+commas?\b|\bnot\s+(use|contain)\s+(any\s+)?commas?\b"
)
_QUOTES = re.compile(r"(?i)подвійн\w+\s+лапк\w+|double\s+quot\w+")
_QUOTES_SCOPE = re.compile(r"(?i)відповід|текст|обгорн|візьм|wrap|response|answer")
# "comma-separated list" asks for commas even if the same sentence says "no spaces after commas".
_WANTS_COMMAS = re.compile(r"(?i)через\s+ком|comma[- ]separated|separated\s+by\s+commas")
_LOWER = re.compile(
    r"(?i)\bмал\w+\s+літер\w*|\bлітери\s+ма\w+\s+бути\s+малими|\bбез\s+(використання\s+)?велик\w+\s+літер\w*"
    r"|\blower\s?case\b|\bno\s+capital\s+letters\b"
)
_UPPER = re.compile(r"(?i)\bвелик\w+\s+літер\w*|\b(all\s+)?capital\s+letters\b|\bupper\s?case\b")
# "at least 3 words in capital letters" is a count, not a whole-answer rule.
_COUNTED = re.compile(r"(?i)щонайменше|принаймні|менше|більше|разів|раз[иа]?\b|\bслів\b|\bслова\b|\bслово\b|\bфраз|at least|times|words")
_WHOLE = re.compile(
    r"(?i)\bвс(я|ю|і|іх|іма|ій|е)\b|\bус(я|ю|і|іх|іма|ій|е)\b|\bлише\b|\bтільки\b|\bповністю\b|відповід|написан|entire|whole|only|all"
)


def detect(request: str) -> list[str]:
    """Constraints stated in the request, in the order they are applied."""
    found: set[str] = set()
    for match in _SENTENCE.finditer(request or ""):
        sentence = match.group(0)
        if _NO_COMMA.search(sentence) and not _WANTS_COMMAS.search(sentence):
            found.add("no_comma")
        if _QUOTES.search(sentence) and _QUOTES_SCOPE.search(sentence):
            found.add("quotes")
        counted = bool(_COUNTED.search(sentence))
        # A count in the same sentence ("the letter g at least 8 times") does not weaken a
        # lowercase rule, so only the capital-letter rule is guarded by `counted`.
        if _LOWER.search(sentence):
            found.add("lower")
        if _UPPER.search(sentence) and not _LOWER.search(sentence) and _WHOLE.search(sentence) and not counted:
            found.add("upper")
    if {"lower", "upper"} <= found:      # contradictory reading: change nothing
        found -= {"lower", "upper"}
    return [name for name in ("lower", "upper", "no_comma", "quotes") if name in found]


def apply(request: str, answer: str) -> tuple[str, list[str]]:
    """Repair `answer` for the constraints in `request`. Returns (answer, repairs made)."""
    text = answer or ""
    made: list[str] = []
    for name in detect(request):
        if name == "lower":
            fixed = text.lower()
        elif name == "upper":
            fixed = text.upper()
        elif name == "no_comma":
            fixed = re.sub(r"\s*,\s*", " ", text) if "," in text else text
        else:
            core = text.strip()
            while len(core) >= 2 and core[0] in "\"“«„" and core[-1] in "\"”»“":
                core = core[1:-1].strip()
            fixed = f'"{core}"'
        if fixed != text:
            made.append(name)
            text = fixed
    return text, made
