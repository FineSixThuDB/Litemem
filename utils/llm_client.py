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
import time
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
        self.usage_callback = None
        api_key = resolve_openai_api_key(self.config.api_key)
        base_url = (
            self.config.base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout)

    @staticmethod
    def _usage_get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _extract_usage(cls, usage: Any) -> Dict[str, Any]:
        prompt_details = cls._usage_get(usage, "prompt_tokens_details", {}) or {}
        completion_details = cls._usage_get(usage, "completion_tokens_details", {}) or {}
        input_details = cls._usage_get(usage, "input_tokens_details", {}) or {}
        output_details = cls._usage_get(usage, "output_tokens_details", {}) or {}
        prompt_tokens = cls._usage_get(
            usage, "prompt_tokens", cls._usage_get(usage, "input_tokens", 0)
        ) or 0
        completion_tokens = cls._usage_get(
            usage, "completion_tokens", cls._usage_get(usage, "output_tokens", 0)
        ) or 0
        cached_tokens = (
            cls._usage_get(prompt_details, "cached_tokens", None)
            if prompt_details
            else cls._usage_get(input_details, "cached_tokens", None)
        )
        reasoning_tokens = (
            cls._usage_get(completion_details, "reasoning_tokens", None)
            if completion_details
            else cls._usage_get(output_details, "reasoning_tokens", None)
        )
        total_tokens = cls._usage_get(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        return {
            "chat_input_tokens": int(prompt_tokens or 0),
            "chat_output_tokens": int(completion_tokens or 0),
            "cached_tokens": int(cached_tokens or 0),
            "reasoning_tokens": int(reasoning_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "usage_missing": usage is None,
        }

    def _emit_usage(
        self,
        *,
        stage: Optional[str],
        usage: Any,
        latency_s: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.usage_callback is None:
            return
        event = {
            "stage": stage or "chat.completion",
            "kind": "chat",
            "model": self.config.model,
            "latency_s": latency_s,
            "embedding_tokens": 0,
            **self._extract_usage(usage),
        }
        if extra:
            event["extra"] = extra
        self.usage_callback(event)

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
        usage_stage: Optional[str] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Return the assistant's reply text.

        Args:
            messages: list of ``{"role": ..., "content": ...}`` dicts.
            response_format: e.g. ``{"type": "json_object"}`` to force JSON.
        """
        params = self._build_params(messages, response_format, **kwargs)
        start = time.perf_counter()
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
        latency_s = time.perf_counter() - start
        self._emit_usage(
            stage=usage_stage,
            usage=getattr(response, "usage", None),
            latency_s=latency_s,
            extra=usage_extra,
        )
        return response.choices[0].message.content or ""
