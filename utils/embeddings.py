"""OpenAI-compatible embeddings client.

Single backend — same idea as ``llm_client.py``. Works against any service
exposing ``POST /v1/embeddings`` in the OpenAI schema.

mem0 ships 14+ embedding providers; LiteMem keeps one. Change models/dims
via ``EmbedderConfig``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from litemem.config import EmbedderConfig
from litemem.utils.auth import resolve_openai_api_key

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """OpenAI-compatible embedding wrapper.

    Vectors are returned as plain ``List[float]``. Batch requests chunk into
    ``config.batch_size`` (default 100) to stay under provider limits.

    The ``dimensions`` parameter is only sent when the user explicitly set
    ``embedding_dims`` to something other than the OpenAI default (1536 for
    text-embedding-3-small). Many OpenAI-compatible backends reject the
    parameter, so we keep mem0's "only send when user opts in" behavior.
    """

    def __init__(self, config: Optional[EmbedderConfig] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        self.config = config or EmbedderConfig()
        api_key = resolve_openai_api_key(self.config.api_key)
        base_url = (
            self.config.base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        # text-embedding-3-small defaults to 1536 on OpenAI. Other OpenAI-compatible
        # providers, e.g. DashScope text-embedding-v4, may need dimensions=1536
        # sent explicitly, so key off both model and dim.
        self._pass_dimensions = (
            self.config.embedding_dims is not None
            and (
                self.config.embedding_dims != 1536
                or (self.config.model or "") != "text-embedding-3-small"
            )
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout)

    def _embed_request_kwargs(self, inputs: List[str]) -> dict:
        kwargs = {
            "model": self.config.model,
            "input": inputs,
            "encoding_format": "float",
        }
        if self._pass_dimensions:
            kwargs["dimensions"] = self.config.embedding_dims
        return kwargs

    def embed(self, text: str, memory_action: Optional[str] = None) -> List[float]:
        """Embed one string.

        ``memory_action`` is accepted for API compatibility with mem0
        (``"add"`` / ``"search"`` / ``"update"``) but ignored — OpenAI's
        embedding API has no per-action variants.
        """
        text = (text or "").replace("\n", " ")
        kwargs = self._embed_request_kwargs([text])
        try:
            response = self.client.embeddings.create(**kwargs)
        except Exception as exc:
            if "dimensions" not in str(exc).lower() or "dimensions" not in kwargs:
                raise
            kwargs.pop("dimensions", None)
            response = self.client.embeddings.create(**kwargs)
        return list(response.data[0].embedding)

    def embed_batch(self, texts: List[str], memory_action: str = "add") -> List[List[float]]:
        """Embed a list of strings, automatically chunking by ``batch_size``."""
        if not texts:
            return []
        cleaned = [(t or "").replace("\n", " ") for t in texts]
        all_embeddings: List[List[float]] = []
        for i in range(0, len(cleaned), self.config.batch_size):
            chunk = cleaned[i: i + self.config.batch_size]
            kwargs = self._embed_request_kwargs(chunk)
            try:
                response = self.client.embeddings.create(**kwargs)
            except Exception as exc:
                if "dimensions" not in str(exc).lower() or "dimensions" not in kwargs:
                    raise
                kwargs.pop("dimensions", None)
                response = self.client.embeddings.create(**kwargs)
            # OpenAI guarantees ordering by .index, but defensively sort.
            ordered = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend(list(item.embedding) for item in ordered)
        return all_embeddings
