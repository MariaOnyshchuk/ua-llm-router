"""vLLM specialist endpoints (same ports as cluster eval / LiteLLM)."""

from __future__ import annotations

BACKENDS: dict[str, tuple[str, str]] = {
    "mamay4": (
        "http://127.0.0.1:8003/v1",
        "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
    ),
    "qwen": (
        "http://127.0.0.1:8004/v1",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
    ),
    "qwen7": (
        "http://127.0.0.1:8004/v1",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
    "mamay12": (
        "http://127.0.0.1:8002/v1",
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
    ),
    "lapa": (
        "http://127.0.0.1:8001/v1",
        "lapa-llm/lapa-v0.1.2-instruct",
    ),
    "aya": (
        "http://127.0.0.1:8005/v1",
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
