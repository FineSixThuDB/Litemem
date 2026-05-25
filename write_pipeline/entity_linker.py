"""Entity linker — Phase 7 of mem0 V3 add().

Responsibility (Phase 7 ↔ ``link_memories`` here):

1. Run spaCy entity extraction on every freshly persisted memory's text
   (one ``nlp.pipe`` batch call).
2. Globally deduplicate entities across the batch
   (key = ``entity_text.strip().lower()``) and collect the set of
   memory_ids that mention each entity.
3. Embed every unique entity in one batch.
4. For each entity, search the entity store (top_k=1, threshold ≥ 0.95)
   to decide between UPDATE (merge ``linked_memory_ids``) vs INSERT.
5. Apply updates one-by-one (must change a list field, not just embedding)
   and apply inserts as one batched ``insert``.

For ``update`` and ``delete`` on a single memory we also expose:

- ``remove_memory_from_entities`` — strip a memory_id from every entity
  that linked to it; drop the entity record if it becomes orphaned.
- ``link_single_memory`` — re-extract entities for one updated memory.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from litemem.data_models import VectorRecord
from litemem.storage.entity_store import EntityStore
from litemem.utils.embeddings import OpenAIEmbedder
from litemem.utils.text_utils import extract_entities, extract_entities_batch

logger = logging.getLogger(__name__)

# Same threshold mem0 uses for "this is the same entity".
ENTITY_DUP_THRESHOLD = 0.95


class EntityLinker:
    def __init__(self, entity_store: EntityStore, embedder: OpenAIEmbedder):
        self.entity_store = entity_store
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Batch path (used by add())
    # ------------------------------------------------------------------

    def link_memories(
        self,
        memory_records: List[Tuple[str, str]],
        *,
        filters: Dict[str, Any],
    ) -> None:
        """Phase 7 — link entities for every memory in the freshly persisted batch.

        Args:
            memory_records: list of ``(memory_id, text)`` tuples for memories
                that were just inserted in Phase 6.
            filters: session-scope filters (user_id/agent_id/run_id) to attach
                to new entity records.
        """
        if not memory_records:
            return

        session_filters = self._session_filters(filters)
        texts = [t for _, t in memory_records]

        try:
            entities_per_text = extract_entities_batch(texts)
        except Exception as e:
            logger.warning(f"Batch entity extraction failed: {e}")
            return

        # 7a: Global dedup across the batch.
        global_entities: Dict[str, List[Any]] = {}  # key -> [entity_type, entity_text, set(mem_id)]
        for idx, (memory_id, _) in enumerate(memory_records):
            entities = entities_per_text[idx] if idx < len(entities_per_text) else []
            for entity_type, entity_text in entities:
                key = entity_text.strip().lower()
                if not key:
                    continue
                if key in global_entities:
                    global_entities[key][2].add(memory_id)
                else:
                    global_entities[key] = [entity_type, entity_text, {memory_id}]

        if not global_entities:
            return

        ordered_keys = list(global_entities.keys())
        entity_texts = [global_entities[k][1] for k in ordered_keys]

        # 7b: Batch-embed all unique entities (with per-item fallback).
        try:
            entity_vectors = self.embedder.embed_batch(entity_texts, "add")
        except Exception:
            entity_vectors = []
            for t in entity_texts:
                try:
                    entity_vectors.append(self.embedder.embed(t, "add"))
                except Exception:
                    entity_vectors.append(None)

        valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_vectors[i] is not None]
        if not valid:
            return
        valid_idx, valid_keys = zip(*valid)
        valid_vectors = [entity_vectors[i] for i in valid_idx]

        # 7c: Search the entity store one-by-one (no native batch in DuckDB).
        existing_matches = self.entity_store.search_batch(
            list(valid_vectors), top_k=1, filters=session_filters
        )

        # 7d: Split into updates vs inserts.
        to_insert_vectors: List[List[float]] = []
        to_insert_ids: List[str] = []
        to_insert_payloads: List[Dict[str, Any]] = []
        for j, key in enumerate(valid_keys):
            entity_type, entity_text, memory_ids = global_entities[key]
            matches = existing_matches[j] if j < len(existing_matches) else []
            top = matches[0] if matches else None
            if top is not None and (top.score or 0.0) >= ENTITY_DUP_THRESHOLD:
                payload = dict(top.payload or {})
                linked = set(payload.get("linked_memory_ids", []))
                linked |= memory_ids
                payload["linked_memory_ids"] = sorted(linked)
                try:
                    self.entity_store.update(top.id, vector=None, payload=payload)
                except Exception as e:
                    logger.debug(f"Entity update failed for '{entity_text}': {e}")
            else:
                to_insert_vectors.append(valid_vectors[j])
                to_insert_ids.append(str(uuid.uuid4()))
                to_insert_payloads.append(
                    {
                        "data": entity_text,
                        "entity_type": entity_type,
                        "linked_memory_ids": sorted(memory_ids),
                        **session_filters,
                    }
                )

        # 7e: Single batch insert for new entities.
        if to_insert_vectors:
            try:
                self.entity_store.insert(
                    vectors=to_insert_vectors,
                    ids=to_insert_ids,
                    payloads=to_insert_payloads,
                )
            except Exception as e:
                logger.warning(f"Batch entity insert failed: {e}")

    # ------------------------------------------------------------------
    # Single-memory path (used by update() / delete())
    # ------------------------------------------------------------------

    def link_single_memory(
        self,
        memory_id: str,
        text: str,
        *,
        filters: Dict[str, Any],
    ) -> None:
        try:
            entities = extract_entities(text)
        except Exception as e:
            logger.warning(f"Entity extraction failed for memory {memory_id}: {e}")
            return
        if not entities:
            return
        session_filters = self._session_filters(filters)
        seen = set()
        for entity_type, entity_text in entities:
            key = entity_text.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            try:
                self._upsert_one(entity_text, entity_type, memory_id, session_filters)
            except Exception as e:
                logger.debug(f"Entity link failed for '{entity_text}': {e}")

    def remove_memory_from_entities(
        self,
        memory_id: str,
        *,
        filters: Dict[str, Any],
    ) -> None:
        session_filters = self._session_filters(filters)
        try:
            rows = self.entity_store.list(filters=session_filters, top_k=10000)
        except Exception as e:
            logger.warning(f"Entity store list failed: {e}")
            return

        for row in rows:
            try:
                payload = dict(row.payload or {})
                linked = payload.get("linked_memory_ids", [])
                if memory_id not in linked:
                    continue
                remaining = [mid for mid in linked if mid != memory_id]
                if not remaining:
                    self.entity_store.delete(row.id)
                else:
                    entity_text = payload.get("data")
                    if not isinstance(entity_text, str) or not entity_text:
                        continue
                    try:
                        vec = self.embedder.embed(entity_text, "update")
                    except Exception as e:
                        logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                        continue
                    payload["linked_memory_ids"] = remaining
                    self.entity_store.update(row.id, vector=vec, payload=payload)
            except Exception as e:
                logger.debug(f"Entity cleanup failure: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _upsert_one(
        self,
        entity_text: str,
        entity_type: str,
        memory_id: str,
        session_filters: Dict[str, Any],
    ) -> None:
        try:
            vec = self.embedder.embed(entity_text, "add")
        except Exception as e:
            logger.debug(f"Entity embed failed for '{entity_text}': {e}")
            return

        existing = self.entity_store.search(vec, top_k=1, filters=session_filters)
        if existing and (existing[0].score or 0.0) >= ENTITY_DUP_THRESHOLD:
            match = existing[0]
            payload = dict(match.payload or {})
            linked = set(payload.get("linked_memory_ids", []))
            if memory_id not in linked:
                linked.add(memory_id)
                payload["linked_memory_ids"] = sorted(linked)
                self.entity_store.update(match.id, vector=None, payload=payload)
        else:
            self.entity_store.insert(
                vectors=[vec],
                ids=[str(uuid.uuid4())],
                payloads=[
                    {
                        "data": entity_text,
                        "entity_type": entity_type,
                        "linked_memory_ids": [memory_id],
                        **session_filters,
                    }
                ],
            )

    @staticmethod
    def _session_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in (filters or {}).items()
            if k in ("user_id", "agent_id", "run_id") and v
        }
