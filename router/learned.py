"""Embedding kNN / logistic-regression router (Hybrid-LLM style).

Default production routing stays rules v2. Profiles `knn` and `clf` load
artifacts from router/artifacts/<kind>/ when present.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from router.intent_rules import ROUTER_MODELS, RouteDecision

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
ALIASES = ("mamay4", "lapa", "aya", "qwen7")


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEncoder:
    """Deterministic stub encoder for tests (no network, no torch)."""

    name = "hash"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            for j in range(self.dim):
                out[i, j] = digest[j % len(digest)]
            norm = float(np.linalg.norm(out[i])) or 1.0
            out[i] /= norm
        return out


class SentenceTransformerEncoder:
    name = "e5"

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        self.model = SentenceTransformer(model_name, **kwargs)
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" if "e5" in self.name.lower() else t for t in texts]
        vecs = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def load_encoder(config: dict[str, Any]) -> Encoder:
    kind = str(config.get("encoder") or "hash")
    if kind == "hash":
        return HashEncoder(int(config.get("encoder_dim") or 32))
    name = str(config.get("encoder_name") or "intfloat/multilingual-e5-large")
    revision = config.get("encoder_revision")
    return SentenceTransformerEncoder(name, revision=str(revision) if revision else None)


def artifact_dir(kind: str) -> Path:
    return ARTIFACT_ROOT / kind


def load_config(kind: str) -> dict[str, Any] | None:
    path = artifact_dir(kind) / "config.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sklearn(kind: str) -> Any:
    import pickle

    path = artifact_dir(kind) / "model.pkl"
    if not path.is_file():
        return None
    return pickle.loads(path.read_bytes())


def predict_alias(text: str, kind: str) -> tuple[str, str]:
    config = load_config(kind)
    if not config:
        return "", f"learned {kind}: no artifacts"
    model = _load_sklearn(kind)
    if model is None:
        return "", f"learned {kind}: missing model.pkl"
    encoder = load_encoder(config)
    vec = encoder.encode([text or ""])
    pred = model.predict(vec)[0]
    alias = str(pred)
    if alias not in ROUTER_MODELS:
        alias = str(config.get("default") or "aya")
    return alias, f"learned {kind} → {alias}"


def route_learned(
    text: str,
    *,
    kind: str,
    default: str = "aya",
    pool: frozenset[str] | set[str] | None = None,
) -> RouteDecision:
    alias, reason = predict_alias(text, kind)
    if not alias:
        alias = default
        reason = f"{reason}; fallback {default}"
    if pool:
        allowed = {m for m in pool if m in ROUTER_MODELS}
        if allowed and alias not in allowed:
            alias = next((m for m in ALIASES if m in allowed), alias)
            reason = f"{reason}; pool → {alias}"
    return RouteDecision(alias, "learned", reason)
