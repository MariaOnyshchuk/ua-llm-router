"""S3 ensemble: majority vote on discrete labels only.

Voters: Lapa, Mamay-4B, Aya. Used for alignment (digit 0–2) and ZNO letters.
Open knowledge / chat / code / translate are not voted.
Tie-break: the rules-v2 first hop.
"""

from __future__ import annotations

import re
from collections import Counter

from router.cascade import ALIGN_DIGIT_RE, ZNO_LETTER_RE

ENSEMBLE_VOTERS = ("lapa", "mamay4", "aya")

_LATIN_TO_CYR = {"a": "а", "b": "б", "c": "в", "d": "г", "e": "д"}


def should_ensemble_vote(item: dict) -> bool:
    bucket = str(item.get("bucket") or "")
    iid = str(item.get("id") or "")
    if bucket == "alignment":
        return True
    if bucket == "knowledge" and iid.startswith("zno-"):
        return True
    return False


def extract_vote_label(item: dict, content: str) -> str | None:
    """Same letter/digit parse as scripts/score_results.py (vote must match scoring)."""
    bucket = str(item.get("bucket") or "")
    text = content or ""
    if bucket == "alignment":
        m = ALIGN_DIGIT_RE.search(text)
        return m.group(1) if m else None
    if bucket == "knowledge":
        raw = text.strip()
        m = ZNO_LETTER_RE.search(raw)
        if not m:
            m = re.search(r"(?:відповідь|answer)\s*[:\-–]?\s*([А-ДA-Ea-eабвгд])\b", raw, re.I)
        if not m:
            m = re.search(r"(?m)^\s*([А-ДA-Ea-eабвгд])\s*[).]?\s*$", raw)
        if not m:
            m = re.search(r"\b([А-ДA-Ea-eабвгд])\b", raw)
        if not m:
            return None
        ch = m.group(1)
        return _LATIN_TO_CYR.get(ch.lower(), ch.lower())
    return None


def majority_winner(
    ballots: list[tuple[str, str | None, dict]],
    tie_alias: str,
) -> tuple[str, str | None, dict]:
    """ballots: (alias, label, chat_result). Returns winning alias, label, result."""
    by_alias = {a: (lab, res) for a, lab, res in ballots}
    counted = [(a, lab) for a, lab, _ in ballots if lab]
    if not counted:
        lab, res = by_alias.get(tie_alias, (None, ballots[0][2]))
        return tie_alias, lab, res

    freq = Counter(lab for _, lab in counted)
    top_lab, top_n = freq.most_common(1)[0]
    second_n = freq.most_common(2)[1][1] if len(freq) > 1 else 0
    if top_n == second_n:
        lab, res = by_alias[tie_alias]
        return tie_alias, lab, res

    if tie_alias in by_alias and by_alias[tie_alias][0] == top_lab:
        return tie_alias, top_lab, by_alias[tie_alias][1]
    for a, lab, res in ballots:
        if lab == top_lab:
            return a, top_lab, res
    lab, res = by_alias[tie_alias]
    return tie_alias, lab, res
