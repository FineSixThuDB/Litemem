"""Data models for LiteMem.

mem0 uses Pydantic. LiteMem uses ``@dataclass`` to keep the data shapes
obvious — every field, every default, and every type is visible at a glance.

The shapes here mirror what mem0's internal code passes between layers:

- ``VectorRecord``   — one row coming back from the vector store
  (has ``id``, ``score``, ``payload`` — same fields the mem0 code reads off
  Qdrant's ``PointStruct``).
- ``ScoredMemory``   — intermediate hit produced by the read pipeline after
  rank fusion; pre-format.
- ``MemoryItem``     — the user-facing dict returned by ``LiteMem.search/get``.
- ``ExtractedFact``  — one entry of the LLM's ``{"memory": [...]}`` output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorRecord:
    """A single row returned by the vector store."""

    id: str
    score: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredMemory:
    """A candidate after rank fusion but before user-facing formatting."""

    id: str
    score: float
    payload: Dict[str, Any]


@dataclass
class ExtractedFact:
    """One memory entry from the LLM extraction JSON."""

    id: str  # sequential integer-as-string ("0", "1", ...)
    text: str
    attributed_to: Optional[str] = None
    linked_memory_ids: List[str] = field(default_factory=list)


@dataclass
class MemoryItem:
    """User-facing memory record.

    Used by LiteMem.search/get/get_all return values. Use ``to_dict()`` to
    convert to a plain dict (drops ``None`` so we don't leak Optional fields).
    """

    id: str
    memory: str
    hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    actor_id: Optional[str] = None
    role: Optional[str] = None

    def to_dict(self, drop_none: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if drop_none:
            return {k: v for k, v in d.items() if v is not None}
        return d


# ---------------------------------------------------------------------------
# Promoted payload keys (shared by storage + context_builder)
# ---------------------------------------------------------------------------

PROMOTED_PAYLOAD_KEYS = ("user_id", "agent_id", "run_id", "actor_id", "role")

CORE_PAYLOAD_KEYS = frozenset(
    {
        "data",
        "hash",
        "created_at",
        "updated_at",
        "id",
        "text_lemmatized",
        "attributed_to",
        "linked_memory_ids",
        *PROMOTED_PAYLOAD_KEYS,
    }
)
