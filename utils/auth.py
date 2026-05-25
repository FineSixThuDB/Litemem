"""Local credential helpers."""

from __future__ import annotations

import json
import os
from typing import Optional


def resolve_openai_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    """Resolve an OpenAI API key from config, env, then Codex auth storage."""
    if explicit_api_key:
        return explicit_api_key

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    auth_path = os.path.expanduser("~/.codex/auth.json")
    try:
        with open(auth_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return None

    key = payload.get("OPENAI_API_KEY")
    return key if isinstance(key, str) and key.strip() else None
