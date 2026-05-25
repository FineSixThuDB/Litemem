"""OpenAI-compatible chat completion client.

Single backend — anything that speaks the OpenAI ``/v1/chat/completions``
schema works: api.openai.com, OpenRouter, vLLM, Together, DeepSeek, Groq, xAI,
Ollama (with ``/v1`` suffix), LM Studio, Azure OpenAI (with right base_url), etc.

mem0 ships 20+ LLM provider files; LiteMem keeps one. To talk to a different
backend, set ``LLMConfig.base_url`` and ``LLMConfig.api_key``.

Reasoning-model handling is included so ``o1`` / ``o3`` / ``gpt-5`` work
without parameters they reject (temperature/top_p/max_tokens).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from litemem.config import LLMConfig
from litemem.utils.auth import resolve_openai_api_key

logger = logging.getLogger(__name__)


class OpenAILLM:
    """OpenAI-compatible chat completion wrapper."""

    # Models that reject standard sampling params (temperature/top_p/max_tokens).
    _REASONING_MODEL_NAMES = {
        "o1", "o1-preview", "o3-mini", "o3",
        "gpt-5", "gpt-5o", "gpt-5o-mini", "gpt-5o-micro",
    }
    _REASONING_MODEL_PREFIXES = ("o1-", "o1.", "o3-", "o3.")

    def __init__(self, config: Optional[LLMConfig] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        self.config = config or LLMConfig()
        api_key = resolve_openai_api_key(self.config.api_key)
        base_url = (
            self.config.base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_reasoning_model(self) -> bool:
        model = (self.config.model or "").lower()
        base = model.rsplit("/", 1)[-1]
        if base in self._REASONING_MODEL_NAMES:
            return True
        return any(base.startswith(p) for p in self._REASONING_MODEL_PREFIXES)

    def _build_params(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if not self._is_reasoning_model():
            params["temperature"] = self.config.temperature
            params["max_tokens"] = self.config.max_tokens
            params["top_p"] = self.config.top_p
        if response_format is not None:
            params["response_format"] = response_format
        if "qwen3" in (self.config.model or "").lower():
            extra_body = dict(params.get("extra_body") or {})
            extra_body.setdefault("enable_thinking", False)
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            params["extra_body"] = extra_body
        params.update(extra)
        return params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Return the assistant's reply text.

        Args:
            messages: list of ``{"role": ..., "content": ...}`` dicts.
            response_format: e.g. ``{"type": "json_object"}`` to force JSON.
        """
        params = self._build_params(messages, response_format, **kwargs)
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            # Some OpenAI-compatible providers reject response_format. Retry once
            # without it so JSON-mode-incompatible backends still work.
            if response_format is not None and "response_format" in str(e):
                logger.warning("Backend rejected response_format; retrying without it.")
                params.pop("response_format", None)
                response = self.client.chat.completions.create(**params)
            else:
                raise
        return response.choices[0].message.content or ""
