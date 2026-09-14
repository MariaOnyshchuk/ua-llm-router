#!/usr/bin/env python3
"""Train embedding kNN and logistic-regression routers from specialist scores.

Labels = argmax quality across specialists. Hold out mixed_ua_v6 ids to avoid leakage.

  PYTHONPATH=. python scripts/train_learned_router.py \\
    --scores results/v6_screen/*_detail.jsonl \\
    --suite benchmarks/mixed_ua_v6_screen.jsonl \\
    --holdout benchmarks/mixed_ua_v6.jsonl \\
    --encoder hash
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.learned import (  # noqa: E402
    ALIASES,
    ARTIFACT_ROOT,
    HashEncoder,
    load_encoder,
)
from router.intent_rules import route_intent  # noqa: E402

SPECIALISTS = {"mamay4", "mamay12", "lapa", "aya", "qwen7", "qwen"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def load_items(path: Path) -> dict[str, dict]:
    return {str(r.get("id")): r for r in load_jsonl(path) if r.get("id")}


def collect_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    by_item: dict[str, dict[str, float]] = defaultdict(dict)
    for path in paths:
        for row in load_jsonl(path):
            sys = str(row.get("system") or "")
            iid = str(row.get("id") or "")
            if sys not in SPECIALISTS or not iid:
                continue
            by_item[iid][sys] = float(row.get("score") or 0.0)
    return by_item


def labels_for(
    items: dict[str, dict],
    scores: dict[str, dict[str, float]],
    *,
    holdout: set[str],
    allowed_aliases: tuple[str, ...],
) -> list[tuple[str, str, str, str]]:
    """Return (id, prompt, alias, bucket) training triples."""
    triples = []
    prefer = list(allowed_aliases)
    for iid, item in items.items():
        if iid in holdout:
            continue
        sc = {k: v for k, v in scores.get(iid, {}).items() if k in allowed_aliases}
        if len(sc) < 2:
            continue
        best = max(sc.values())
        winners = [a for a in prefer if sc.get(a) == best]
        if not winners:
            continue
        prompt = str(item.get("prompt") or "")
        if not prompt:
            continue
        triples.append((iid, prompt, winners[0], str(item.get("bucket") or "")))
    return triples


def fit_and_save(kind: str, X, y, config: dict, encoder_name: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    if kind == "knn":
        model = KNeighborsClassifier(n_neighbors=min(5, max(1, len(set(y)))), weights="distance")
    else:
        model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X, y)
    dest = ARTIFACT_ROOT / kind
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.pkl").write_bytes(pickle.dumps(model))
    cfg = {
        **config,
        "kind": kind,
        "encoder": encoder_name,
        "n_train": int(len(y)),
        "labels": sorted(set(y)),
    }
    (dest / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    acc = float(model.score(X, y))
    return {"kind": kind, "train_acc": round(acc, 4), "n_train": len(y), "wrote": str(dest)}


def lobo(
    triples: list[tuple[str, str, str, str]],
    encode,
    *,
    kinds: tuple[str, ...] = ("knn", "clf"),
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    buckets = sorted({t[3] for t in triples if t[3]})
    out: dict[str, dict] = {}
    for held in buckets:
        train = [t for t in triples if t[3] != held]
        test = [t for t in triples if t[3] == held]
        if len(train) < 8 or len(test) < 4:
            out[held] = {"skipped": True, "n_train": len(train), "n_test": len(test)}
            continue
        Xtr = encode([t[1] for t in train])
        ytr = [t[2] for t in train]
        Xte = encode([t[1] for t in test])
        yte = [t[2] for t in test]
        out[held] = {"n_train": len(train), "n_test": len(test)}
        for kind in kinds:
            if kind == "knn":
                m = KNeighborsClassifier(n_neighbors=min(5, max(1, len(set(ytr)))), weights="distance")
            else:
                m = LogisticRegression(max_iter=500, class_weight="balanced")
            m.fit(Xtr, ytr)
            out[held][kind] = round(float(m.score(Xte, yte)), 4)
        rules_ok = 0
        for _iid, prompt, gold, _b in test:
            pred = route_intent(prompt, profile="v2").model
            rules_ok += int(pred == gold)
        out[held]["rules_v2"] = round(rules_ok / max(1, len(test)), 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", nargs="+", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--holdout", type=Path, default=None)
    p.add_argument("--encoder", default="hash", choices=["hash", "e5", "labse"])
    p.add_argument("--encoder-name", default="")
    p.add_argument("--encoder-revision", default="")
    args = p.parse_args()

    items = load_items(args.suite)
    holdout = set(load_items(args.holdout)) if args.holdout and args.holdout.exists() else set()
    scores = collect_scores(args.scores)
    allowed = tuple(a for a in ALIASES)
    triples = labels_for(items, scores, holdout=holdout, allowed_aliases=allowed)
    if len(triples) < 8:
        raise SystemExit(f"need ≥8 labeled items, got {len(triples)}")

    encoder_name = args.encoder
    config: dict = {"default": "aya", "encoder": args.encoder, "encoder_dim": 32}
    if args.encoder == "e5":
        config["encoder_name"] = args.encoder_name or "intfloat/multilingual-e5-large"
        if args.encoder_revision:
            config["encoder_revision"] = args.encoder_revision
        encoder_name = "e5"
    elif args.encoder == "labse":
        config["encoder_name"] = args.encoder_name or "sentence-transformers/LaBSE"
        encoder_name = "labse"
        config["encoder"] = "labse"

    encoder = load_encoder(config) if args.encoder != "hash" else HashEncoder(32)
    X = encoder.encode([t[1] for t in triples])
    config["encoder_dim"] = int(X.shape[1])
    y = [t[2] for t in triples]
    reports = [
        fit_and_save("knn", X, y, config, encoder_name if args.encoder != "hash" else "hash"),
        fit_and_save("clf", X, y, config, encoder_name if args.encoder != "hash" else "hash"),
    ]
    lobo_report = lobo(triples, encoder.encode)
    payload = {
        "n_labeled": len(triples),
        "n_holdout_ids": len(holdout),
        "label_counts": {a: sum(1 for t in triples if t[2] == a) for a in allowed},
        "models": reports,
        "leave_one_bucket_out": lobo_report,
    }
    summary = ARTIFACT_ROOT / "train_report.json"
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
