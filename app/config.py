"""Configuration helpers for the Doc Chatbot application.

This module defines default settings and provides utilities to load and save
user configuration from the path returned by ``rag.paths()['config']``. Values
are merged with sensible defaults and coerced to the expected types.
"""

import json
import os
from typing import Any

from rag import paths

DEFAULTS: dict[str, Any] = {
    # Provider: 'local' uses free, local embeddings; 'openai' uses OpenAI APIs.
    "provider": "local",
    # OpenAI models (used only if provider == 'openai')
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-small",
    # Local embedding model (used if provider == 'local')
    "embed_model_local": "BAAI/bge-small-en-v1.5",
    # Retrieval parameters
    "top_k": 4,
    "chunk_chars": 1200,
    "overlap": 200,
    "mmr": True,
    "mmr_lambda": 0.5,
    # Hybrid BM25 + embeddings (local provider)
    "hybrid": False,
    "hybrid_weight": 0.5,
}


def load_config() -> dict[str, Any]:
    """Load configuration from disk, merging with defaults.

    The function attempts to read a JSON config file from
    ``rag.paths()['config']``. Any missing or invalid values fall back to
    ``DEFAULTS``. Simple type coercion is applied to ensure consistent types.

    Returns:
        dict[str, object]: The effective configuration mapping.
    """
    p = paths()["config"]
    try:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
    except Exception:
        data = {}
    cfg: dict[str, Any] = DEFAULTS.copy()
    cfg.update({k: data.get(k, v) for k, v in DEFAULTS.items()})
    # type coercion
    cfg["top_k"] = int(cfg["top_k"])
    cfg["chunk_chars"] = int(cfg["chunk_chars"])
    cfg["overlap"] = int(cfg["overlap"])
    cfg["mmr"] = bool(cfg["mmr"])
    cfg["mmr_lambda"] = float(cfg["mmr_lambda"])
    cfg["hybrid"] = bool(cfg.get("hybrid", False))
    cfg["hybrid_weight"] = float(cfg.get("hybrid_weight", 0.5))
    cfg["provider"] = str(cfg.get("provider", "local"))
    cfg["embed_model_local"] = str(cfg.get("embed_model_local", "BAAI/bge-small-en-v1.5"))
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    """Persist configuration to the configured JSON file.

    Args:
        cfg: Mapping of settings to write.
    """
    p = paths()["config"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
