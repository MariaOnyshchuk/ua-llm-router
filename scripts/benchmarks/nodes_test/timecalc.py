"""Clock arithmetic for test items: the gold time is computed, never typed."""

from nodes_train.common import MT


def _minutes(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _clock(minutes: int) -> str:
    minutes %= 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def MTA(sub, form, prompt, start, delta):
    """Clock time `start` plus `delta` minutes (negative to go back); wrong answer is one hour off."""
    gold = _clock(_minutes(start) + delta)
    return MT(sub, form, prompt, gold, _clock(_minutes(start) + delta - 60), f"{start} + {delta} min = {gold}")


def MTS(sub, form, prompt, start, delta_seconds):
    """Clock time with seconds, ГГ:ХХ:СС."""
    h, m, s = map(int, start.split(":"))
    total = (h * 3600 + m * 60 + s + delta_seconds) % 86400
    gold = f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
    return MT(sub, form, prompt, gold, f"{(total // 3600 + 23) % 24:02d}:{total % 3600 // 60:02d}:{total % 60:02d}",
              f"{start} + {delta_seconds} s = {gold}")
