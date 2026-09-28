"""vLLM specialist endpoints (same ports as cluster eval / LiteLLM)."""

from __future__ import annotations

import os


def _v1(port_env: str, default_port: int) -> str:
    """Base URL. Override the port when two jobs share a host."""
    port = os.environ.get(port_env, str(default_port))
    return f"http://127.0.0.1:{port}/v1"


BACKENDS: dict[str, tuple[str, str]] = {
    "mamay4": (
        _v1("ROUTER_PORT_MAMAY4", 8003),
        "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
    ),
    "qwen": (
        _v1("ROUTER_PORT_QWEN", 8004),
        "Qwen/Qwen2.5-Coder-3B-Instruct",
    ),
    "qwen7": (
        _v1("ROUTER_PORT_QWEN7", 8004),
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
    "mamay12": (
        _v1("ROUTER_PORT_MAMAY12", 8002),
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
    ),
    "lapa": (
        _v1("ROUTER_PORT_LAPA", 8001),
        "lapa-llm/lapa-v0.1.2-instruct",
    ),
    "aya": (
        _v1("ROUTER_PORT_AYA", 8005),
        "CohereLabs/aya-expanse-8b",
    ),
}

PLAYGROUND_MODELS = ("mamay4", "lapa", "aya", "qwen7")

MODEL_META: dict[str, dict[str, str]] = {
    "mamay4": {
        "name": "Mamay-4B",
        "lineage": "Gemma-UA · INSAIT",
        "role": "Instruct, UA code",
        "port": "8003",
    },
    "lapa": {
        "name": "Lapa-12B",
        "lineage": "Gemma-UA",
        "role": "Knowledge, alignment",
        "port": "8001",
    },
    "aya": {
        "name": "Aya Expanse 8B",
        "lineage": "Cohere",
        "role": "Translate, default chat",
        "port": "8005",
    },
    "qwen7": {
        "name": "Qwen2.5-Coder-7B",
        "lineage": "Qwen",
        "role": "Code (rules v1)",
        "port": "8004",
    },
    "mamay12": {
        "name": "Mamay-12B",
        "lineage": "Gemma-UA · INSAIT",
        "role": "Legacy cascade target",
        "port": "8002",
    },
}
