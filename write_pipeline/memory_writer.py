"""Memory writer — Phase 3, 6, 8 of mem0 V3 add().

Responsibility:
- **Phase 3** — batch embed every extracted fact's text (with single-item
  fallback if the batch endpoint errors).
- **Phase 6** — assemble payloads (data, hash, text_lemmatized, created_at,
  updated_at, attributed_to, linked_memory_ids, session ids), batch-insert
  into the vector store and batch-write the history records.
- **Phase 8** — persist the new messages to the SQLite messages buffer.

It also exposes ``write_raw_message`` for the ``infer=False`` path (a single
message goes straight in without going through the LLM extractor).
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from litemem.data_models import ExtractedFact
from litemem.storage.memory_store import MemoryStore, build_session_scope
from litemem.storage.vector_store import VexDBVectorStore
from litemem.utils.embeddings import OpenAIEmbedder
from litemem.utils.text_utils import lemmatize_for_bm25, md5_hash

logger = logging.getLogger(__name__)


class MemoryWriter:
    def __init__(
        self,
        vector_store: VexDBVectorStore,
        memory_store: MemoryStore,
        embedder: OpenAIEmbedder,
    ):
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Inference path (Phase 3 + 6 + 8)
    # ------------------------------------------------------------------

    def write_facts(
        self,
        facts: List[ExtractedFact],
        fact_hashes: List[str],
        *,
        metadata: Dict[str, Any],
        messages: List[Dict[str, Any]],
        session_scope: str,
    ) -> List[Tuple[str, str]]:
        """Embed → persist → record history → save messages.

        Returns ``[(memory_id, text), ...]`` of memories actually written
        (so the caller can hand them off to the entity linker).
        """
        if not facts:
            # No facts to write, but we still want to persist the raw messages
            # so the next add() has the right "last_k_messages" context.
            self.memory_store.save_messages(messages, session_scope)
            return []

        # Phase 3 — batch embed.
        texts = [f.text for f in facts]
        try:
            vectors = self.embedder.embed_batch(texts, "add")
            embed_map = dict(zip(texts, vectors))
        except Exception:
            embed_map = {}
            for t in texts:
                try:
                    embed_map[t] = self.embedder.embed(t, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed fact text: {e}")

        # Phase 6a — assemble per-record payloads.
        now = datetime.now(timezone.utc).isoformat()
        records: List[Tuple[str, str, List[float], Dict[str, Any]]] = []
        for fact, h in zip(facts, fact_hashes):
            vec = embed_map.get(fact.text)
            if vec is None:
                continue
            memory_id = str(uuid.uuid4())
            payload = deepcopy(metadata) or {}
            payload["data"] = fact.text
            payload["hash"] = h
            payload["text_lemmatized"] = lemmatize_for_bm25(fact.text)
            payload.setdefault("created_at", now)
            payload["updated_at"] = payload["created_at"]
            if fact.attributed_to:
                payload["attributed_to"] = fact.attributed_to
            if fact.linked_memory_ids:
                payload["linked_memory_ids"] = list(fact.linked_memory_ids)
            records.append((memory_id, fact.text, vec, payload))

        if not records:
            self.memory_store.save_messages(messages, session_scope)
            return []

        # Phase 6b — batch insert into vector store (with single-item fallback).
        try:
            self.vector_store.insert(
                vectors=[r[2] for r in records],
                ids=[r[0] for r in records],
                payloads=[r[3] for r in records],
            )
        except Exception as e:
            logger.warning(f"Batch insert failed ({e}); falling back to per-item insert.")
            for mid, _, vec, payload in records:
                try:
                    self.vector_store.insert(
                        vectors=[vec], ids=[mid], payloads=[payload]
                    )
                except Exception as e2:
                    logger.error(f"Failed to insert memory {mid}: {e2}")

        # Phase 6c — batch history.
        try:
            self.memory_store.batch_add_history(
                [
                    {
                        "memory_id": r[0],
                        "old_memory": None,
                        "new_memory": r[1],
                        "event": "ADD",
                        "created_at": r[3].get("created_at"),
                        "updated_at": r[3].get("updated_at"),
                        "is_deleted": 0,
                        "actor_id": r[3].get("actor_id"),
                        "role": r[3].get("role"),
                    }
                    for r in records
                ]
            )
        except Exception as e:
            logger.warning(f"Batch history failed ({e}); falling back to per-item.")
            for mid, text, _, payload in records:
                try:
                    self.memory_store.add_history(
                        mid, None, text, "ADD",
                        created_at=payload.get("created_at"),
                        updated_at=payload.get("updated_at"),
                        actor_id=payload.get("actor_id"),
                        role=payload.get("role"),
                    )
                except Exception as e2:
                    logger.error(f"Failed to add history for {mid}: {e2}")

        # Phase 8 — persist raw messages into the recent buffer.
        self.memory_store.save_messages(messages, session_scope)

        return [(r[0], r[1]) for r in records]

    # ------------------------------------------------------------------
    # Raw (infer=False) path — used when caller wants to store messages
    # verbatim without LLM extraction.
    # ------------------------------------------------------------------

    def write_raw_message(
        self,
        text: str,
        *,
        metadata: Dict[str, Any],
    ) -> str:
        """Persist a single raw message string as a memory record."""
        vec = self.embedder.embed(text, "add")
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        payload = deepcopy(metadata) or {}
        payload["data"] = text
        payload["hash"] = md5_hash(text)
        payload["text_lemmatized"] = lemmatize_for_bm25(text)
        payload.setdefault("created_at", now)
        payload["updated_at"] = payload["created_at"]
        self.vector_store.insert(vectors=[vec], ids=[memory_id], payloads=[payload])
        self.memory_store.add_history(
            memory_id, None, text, "ADD",
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            actor_id=payload.get("actor_id"),
            role=payload.get("role"),
        )
        return memory_id

    # ------------------------------------------------------------------
    # Update / delete on a single memory
    # ------------------------------------------------------------------

    def update_memory(
        self,
        memory_id: str,
        new_text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        existing = self.vector_store.get(memory_id)
        if existing is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        prev_payload = existing.payload or {}
        prev_text = prev_payload.get("data")

        # Merge metadata, preserving session ids unless overridden.
        new_payload = deepcopy(metadata) if metadata else {}
        for k in ("user_id", "agent_id", "run_id", "actor_id", "role"):
            if k not in new_payload and k in prev_payload and prev_payload[k] is not None:
                new_payload[k] = prev_payload[k]
        new_payload["data"] = new_text
        new_payload["hash"] = md5_hash(new_text)
        new_payload["text_lemmatized"] = lemmatize_for_bm25(new_text)
        new_payload["created_at"] = prev_payload.get("created_at")
        new_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        # Preserve memory-linking metadata if not overwritten.
        if "linked_memory_ids" not in new_payload and prev_payload.get("linked_memory_ids"):
            new_payload["linked_memory_ids"] = prev_payload["linked_memory_ids"]
        if "attributed_to" not in new_payload and prev_payload.get("attributed_to"):
            new_payload["attributed_to"] = prev_payload["attributed_to"]

        vec = self.embedder.embed(new_text, "update")
        self.vector_store.update(memory_id, vector=vec, payload=new_payload)
        self.memory_store.add_history(
            memory_id, prev_text, new_text, "UPDATE",
            created_at=new_payload["created_at"],
            updated_at=new_payload["updated_at"],
            actor_id=new_payload.get("actor_id"),
            role=new_payload.get("role"),
        )
        return memory_id, new_payload

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        existing = self.vector_store.get(memory_id)
        if existing is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        payload = existing.payload or {}
        prev_text = payload.get("data", "")
        self.vector_store.delete(memory_id)
        self.memory_store.add_history(
            memory_id, prev_text, None, "DELETE",
            created_at=payload.get("created_at"),
            updated_at=datetime.now(timezone.utc).isoformat(),
            actor_id=payload.get("actor_id"),
            role=payload.get("role"),
            is_deleted=1,
        )
        return payload
