#!/usr/bin/env python3
"""Build the node datasets: single-skill items of the step types found in graphs_mixed_100_dev.

  train  benchmarks/nodes_mixed_600_train.jsonl  labels for the learned routers (kNN, MLP); items in nodes_train/
  test   benchmarks/nodes_mixed_600_test.jsonl   held out for comparing routers; items in nodes_test/

Items are drafted by Claude (not one of the compared models) following
`scripts/benchmarks/prompts/node_train_generation.md`. Real ZNO items (knowledge) and UAlign social items
(alignment) are drawn here at random, seed 42, from `benchmarks/corpus/`, never items used in
tasks_public_4401, the graph sets or the other node split.

Build checks (an item that fails stops the build), run on the items after a JSON round trip:
- the gold scores 1.0 on its own check, the written wrong answer scores below 1.0;
- math golds are computed from the arithmetic in the item; code golds pass every unit test;
- no written prompt shares a 10-word sequence with tasks_public_4401, graphs_mixed_100_dev or graphs_mixed_100_test;
- no two prompts are the same after normalising spaces and case.
Test split only:
- no written prompt shares a 10-word sequence with, or equals, a train prompt;
- the same subtype x form counts as train in every skill;
- inside a skill no two written items share their first three words, and no four-word phrase is in more than
  three of them (fixed definitions such as the 0/1/2 scale and answer markers excepted);
- no written item is closer than 0.95 (e5 cosine, multilingual-e5-large) to a train item of its skill.

Items are never selected using model scores.

    python scripts/benchmarks/build_nodes_mixed_600.py               # train
    python scripts/benchmarks/build_nodes_mixed_600.py --split test
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SKILLS = ("math", "translate", "instruct", "knowledge", "alignment", "code")
N_ZNO, N_UALIGN = 40, 15

_spec = importlib.util.spec_from_file_location("scorer", ROOT / "scripts" / "score_composite_results.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

from dataset_headers import why_line  # noqa: E402
from nodes_train.common import ROOT as FORM_ROOT, SOC, digit, item, letter  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# Instruction wording that node prompts share with graphs_mixed_100_dev by design (the alignment scale, answer formats,
# the list of Ukrainian vowels). It is removed before the overlap check, which then compares content only.
BOILERPLATE = (
    "Оціни соціальну прийнятність дії за шкалою", "0 — погано, 1 — очікувано (звичайна поведінка), 2 — добре (явно похвально)",
    "Відповідай лише цифрою", 'Відповідай у форматі "Оцінка: <цифра>"',
    'Відповідь дай двома рядками: "Дія: <дія одним реченням>" і "Оцінка: <цифра>"', "а, е, є, и, і, ї, о, у, ю, я",
    "Наприкінці напиши: Відповідь: <число>.",
)


def ngrams(text: str, n: int = 10) -> set[str]:
    for phrase in BOILERPLATE:
        text = text.replace(phrase, " | ")
    words = re.findall(r"\w+|\|", text.casefold())
    return {g for g in (" ".join(words[i:i + n]) for i in range(len(words) - n + 1)) if "|" not in g}


def reference_texts_and_ids() -> tuple[set[str], set[str]]:
    grams: set[str] = set()
    ids: set[str] = set()
    for name in ("tasks_public_4401.jsonl", "graphs_mixed_100_dev.jsonl", "graphs_mixed_100_test.jsonl"):
        for r in load_jsonl(ROOT / "benchmarks" / name):
            ids.add(r["id"])
            for sid in str(r.get("source_id") or "").split(","):
                ids.add(sid.strip())
            grams |= ngrams(r.get("prompt") or "")
            plan = r.get("oracle_plan")
            if isinstance(plan, dict):
                for step in plan.get("steps") or []:
                    grams |= ngrams(re.sub(r"\{\{[^}]+\}\}", " ", step.get("prompt") or ""))
    return grams, ids


def draw_zno(used: set[str], rng: random.Random) -> list[dict]:
    pool = [r for r in load_jsonl(ROOT / "benchmarks/corpus/zno_knowledge_full.jsonl")
            if r["id"].split("-")[1] in ("ukr", "his") and r["id"] not in used
            and len(r["prompt"]) < 900 and r.get("reference") in list("АБВГД")]
    out = []
    for r in rng.sample(pool, N_ZNO):
        sub = "ZNO, Ukrainian language and literature" if "-ukr-" in r["id"] else "ZNO, history of Ukraine"
        wrong = next(x for x in "АБВГД" if x != r["reference"])
        out.append(item("knowledge", "knowledge", sub, FORM_ROOT, r["prompt"], r["reference"], letter(r["reference"]),
                        wrong, does="letter (source gold)", source_id=r["id"]))
    return out


def draw_ualign(used: set[str], rng: random.Random) -> list[dict]:
    pool = [r for r in load_jsonl(ROOT / "benchmarks/corpus/ualign_social_full.jsonl")
            if r["id"] not in used and str(r.get("reference")) in ("0", "1", "2")]
    out = []
    for r in rng.sample(pool, N_UALIGN):
        action = r["prompt"].split("\n\n", 1)[-1].strip()
        value = int(r["reference"])
        out.append(item("alignment", "alignment", "UAlign social, real item", FORM_ROOT,
                        f"{SOC} Відповідай лише цифрою.\nДія: {action}", str(value), digit(value),
                        str(1 if value != 1 else 2), does="digit (source gold)", source_id=r["id"]))
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def opener_and_phrases(prompt: str) -> tuple[str, set[str]]:
    from nodes_train.common import ANS, SCALE
    for fixed in (SCALE, ANS.strip(), "а, е, є, и, і, ї, о, у, ю, я", '"Оцінка: <цифра>"', '"Дія: <дія одним реченням>"'):
        prompt = prompt.replace(fixed, " | ")
    words = re.findall(r"\w+|\|", prompt.casefold())
    phrases = {" ".join(words[i:i + 4]) for i in range(len(words) - 3) if "|" not in words[i:i + 4]}
    return " ".join(words[:3]), phrases


def test_split_errors(items: list[dict], train: list[dict]) -> list[str]:
    errors: list[str] = []
    train_grams: set[str] = set()
    for r in train:
        train_grams |= ngrams(r["prompt"])
    train_norm = {norm(r["prompt"]) for r in train}
    for skill in SKILLS:
        own = [it for it in items if it["skill"] == skill and "source_id" not in it]
        want = Counter((r["subtype"], r["form"]) for r in train if r["skill"] == skill and "source_id" not in r)
        if Counter((it["subtype"], it["form"]) for it in own) != want:
            errors.append(f"{skill}: subtype x form counts differ from train")
        openers: dict[str, str] = {}
        phrase_items: dict[str, list[str]] = {}
        for it in own:
            if ngrams(it["prompt"]) & train_grams:
                errors.append(f"{it['id']}: shares a 10-word sequence with a train prompt")
            if norm(it["prompt"]) in train_norm:
                errors.append(f"{it['id']}: same prompt as a train item")
            opener, phrases = opener_and_phrases(it["prompt"])
            if opener in openers:
                errors.append(f"{it['id']}: same first three words as {openers[opener]}")
            openers[opener] = it["id"]
            for ph in phrases:
                phrase_items.setdefault(ph, []).append(it["id"])
        errors += [f"{skill}: phrase {ph!r} in {len(ids)} items" for ph, ids in phrase_items.items() if len(ids) > 3]
    return errors + similarity_errors(items, train)


def similarity_errors(items: list[dict], train: list[dict], limit: float = 0.95) -> list[str]:
    from nodes_train.common import SCALE
    spec = importlib.util.spec_from_file_location("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)

    def content(r: dict) -> str:  # alignment items share the scale wording, so compare what follows it
        return r["prompt"].split(SCALE, 1)[-1] if r["skill"] == "alignment" else r["prompt"]

    errors = []
    for skill in SKILLS:
        own = [it for it in items if it["skill"] == skill and "source_id" not in it]
        ref = [r for r in train if r["skill"] == skill]
        X = rb.embed([content(r) for r in own], "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
        T = rb.embed([content(r) for r in ref], "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
        S = X @ T.T
        for k, row in enumerate(S):
            if row.max() > limit:
                errors.append(f"{own[k]['id']}: cosine {row.max():.3f} to train {ref[int(row.argmax())]['id']}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="train")
    split = parser.parse_args().split
    out = ROOT / "benchmarks" / f"nodes_mixed_600_{split}.jsonl"
    meta_path = ROOT / "benchmarks" / f"nodes_mixed_600_{split}_meta.json"
    package, prefix = ("nodes_train", "nt") if split == "train" else ("nodes_test", "nx")

    rng = random.Random(42)
    grams, used_ids = reference_texts_and_ids()
    train: list[dict] = []
    if split == "test":
        train = load_jsonl(ROOT / "benchmarks" / "nodes_mixed_600_train.jsonl")
        used_ids |= {r["source_id"] for r in train if "source_id" in r}
    items: list[dict] = []
    for skill in SKILLS:
        module = importlib.import_module(f"{package}.{skill}")
        own = list(module.ITEMS)
        if skill == "knowledge":
            own += draw_zno(used_ids, rng)
        if skill == "alignment":
            own += draw_ualign(used_ids, rng)
        for k, it in enumerate(own, 1):
            items.append({"id": f"{prefix}-{skill}-{k:03d}", "split": split, "bucket": skill, **it})

    # Check items as they will be read back from the JSONL file (a tuple in a test case becomes a list).
    items = [json.loads(json.dumps(it, ensure_ascii=False)) for it in items]
    errors: list[str] = []
    seen: dict[str, str] = {}
    for it in items:
        g, gd = scorer.score_rubric(it["gold"], it["check"])
        w, wd = scorer.score_rubric(it["wrong"], it["check"])
        if g != 1.0:
            errors.append(f"{it['id']}: gold scores {g} ({gd})")
        if w >= 1.0:
            errors.append(f"{it['id']}: wrong answer passes ({wd})")
        key = norm(it["prompt"])
        if key in seen:
            errors.append(f"{it['id']}: same prompt as {seen[key]}")
        seen[key] = it["id"]
        if "source_id" not in it and ngrams(it["prompt"]) & grams:
            errors.append(f"{it['id']}: shares a 10-word sequence with tasks_public_4401 / graphs_mixed_100_dev / graphs_mixed_100_test")
    if split == "test":
        errors += test_split_errors(items, train)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(f"{len(errors)} build errors")

    body = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    fingerprint = hashlib.sha256(body.encode()).hexdigest()[:16]
    head = (f"# nodes_mixed_600_{split} — 100 per skill (math, translate, instruct, knowledge, alignment, code); built by "
            f"scripts/benchmarks/build_nodes_mixed_600.py; fingerprint={fingerprint}; generated, do not hand-edit\n")
    out.write_text(why_line(out.name) + head + body, encoding="utf-8")
    checks = ("gold passes its check, wrong answer fails it, math from arithmetic, code tests, "
              "no 10-word overlap with tasks_public_4401 / graphs_mixed_100_dev / graphs_mixed_100_test, no duplicate prompts")
    if split == "test":
        checks += ("; no overlap with train, same subtype x form counts as train, unique openers and no four-word "
                   "phrase in more than three items per skill, e5 cosine to train at most 0.95")
    meta = {
        "what": ("single-skill node items for training the kNN / MLP routers on graphs_mixed_100_dev step types"
                 if split == "train" else "held-out single-skill node items for comparing the routers with rules v2"),
        "split": split,
        "n_items": len(items),
        "per_skill": dict(Counter(it["skill"] for it in items)),
        "per_form": {s: dict(Counter(it["form"] for it in items if it["skill"] == s)) for s in SKILLS},
        "per_subtype": {s: dict(Counter(it["subtype"] for it in items if it["skill"] == s)) for s in SKILLS},
        "real_items": {"zno": N_ZNO, "ualign_social": N_UALIGN, "seed": 42},
        "drafted_by": "Claude (Anthropic), not one of the compared models; prompt "
                      "scripts/benchmarks/prompts/node_train_generation.md",
        "checks": checks,
        "teacher_check_n2": "not run",
        "human_review": ("draft sample of 43 approved 31 Oct 2026; full set not reviewed yet" if split == "train"
                         else "draft sample of 42 approved 1 Nov 2026; full set not reviewed yet"),
        "selected_by_model_scores": False,
        "sha256_16": fingerprint,
        "git_commit": git_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("split", "n_items", "per_skill", "sha256_16")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
