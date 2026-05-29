"""LiteMem configuration objects.

Plain ``@dataclass`` config — no Pydantic dependency.

A LiteMem instance is configured by composing four sub-configs:
- ``LLMConfig``           — OpenAI-compatible chat completions endpoint
- ``EmbedderConfig``      — OpenAI-compatible embeddings endpoint
- ``VectorStoreConfig``   — VexDB-Lite (DuckDB + VEX) parameters
- ``MemoryConfig``        — history db path, version, custom instructions

``LiteMemConfig`` (and its alias ``MemoryConfig``) is the top-level object
passed to ``LiteMem(config=...)``.

Defaults match mem0 where possible (text-embedding-3-small @ 1536d,
gpt-4o-mini for chat, cosine distance, history.db under ``~/.litemem/``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# Default directory for persistent files (history sqlite, vex .db).
HOME_DIR = os.path.expanduser("~")
LITEMEM_DIR = os.environ.get("LITEMEM_DIR") or os.path.join(HOME_DIR, ".litemem")


@dataclass
class LLMConfig:
    """OpenAI-compatible chat completions config.

    Works with: api.openai.com, OpenRouter, vLLM, Together, DeepSeek, LM Studio,
    Groq, xAI, Ollama (with /v1), or any service exposing
    ``POST /v1/chat/completions`` in the OpenAI schema.
    """

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None  # falls back to OPENAI_API_KEY
    base_url: Optional[str] = None  # falls back to OPENAI_BASE_URL or api.openai.com
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 0.1
    response_format_json: bool = True  # request {"type": "json_object"} on extraction calls
    timeout: float = 60.0


@dataclass
class EmbedderConfig:
    """OpenAI-compatible embeddings config."""

    model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    api_key: Optional[str] = None  # falls back to OPENAI_API_KEY
    base_url: Optional[str] = None  # falls back to OPENAI_BASE_URL or api.openai.com
    timeout: float = 30.0
    batch_size: int = 100  # OpenAI accepts up to 2048; 100 is a safe default


@dataclass
class VectorStoreConfig:
    """VexDB-Lite vector store config.

    VexDB-Lite is a DuckDB extension. Each LiteMem instance writes to one
    .db file (``db_path``) and uses two tables:
      - ``{collection_name}``           — memory rows
      - ``{collection_name}_entities``  — entity rows for boost retrieval

    ``distance_metric`` is forwarded to ``CREATE INDEX ... WITH (metric=...)``.
    """

    collection_name: str = "litemem"
    embedding_dims: int = 1536
    db_path: str = field(default_factory=lambda: os.path.join(LITEMEM_DIR, "litemem.db"))
    distance_metric: str = "cosine"  # 'l2' | 'cosine' | 'ip'
    ef_search: int = 100  # vex_ef_search — higher = more recall
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200


UsageCallback = Callable[[Dict[str, Any]], None]


@dataclass
class TechniqueFlags:
    """Feature switches used by the ablation harness.

    Defaults preserve LiteMem's normal behavior. These flags are intentionally
    narrow and map to the experiment labels in ``litemem/exp.md``.
    """

    use_additive_extraction: bool = True
    use_existing_memory_context: bool = True
    use_recent_messages_context: bool = True
    use_uuid_anonymization: bool = True
    use_json_response_format: bool = True
    use_bm25: bool = True
    use_entity_boost: bool = True
    use_hash_dedup: bool = True


@dataclass
class LiteMemConfig:
    """Top-level LiteMem configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    history_db_path: str = field(default_factory=lambda: os.path.join(LITEMEM_DIR, "history.db"))
    version: str = "v1.1"
    custom_instructions: Optional[str] = None
    # Cap on session-scoped recent messages stored for additive extraction context.
    recent_messages_limit: int = 10
    # Cap on existing memories retrieved during add() for the LLM to dedupe against.
    existing_memories_limit: int = 10
    # Experiment-only knobs. Defaults keep the normal LiteMem behavior.
    technique_flags: TechniqueFlags = field(default_factory=TechniqueFlags)
    usage_callback: Optional[UsageCallback] = None

    def __post_init__(self):
        os.makedirs(LITEMEM_DIR, exist_ok=True)
        # Propagate dims from embedder to vector_store if user only set one.
        if self.vector_store.embedding_dims != self.embedder.embedding_dims:
            # Embedder is the source of truth; sync vector_store to match.
            self.vector_store.embedding_dims = self.embedder.embedding_dims


# Alias matching the name mem0 users expect.
MemoryConfig = LiteMemConfig
