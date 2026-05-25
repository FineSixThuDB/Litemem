"""Entity retriever — Step 6 of mem0 V3 search (entity boost).

For every entity extracted from the query (deduped, max 8):
1. Embed the entity text.
2. Vector-search the entity store within the same session scope (top_k=500).
3. For each match with similarity ≥ 0.5, add a boost to every memory in
   its ``linked_memory_ids`` list.

The boost formula matches mem0 exactly:

    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
    memory_count_weight = 1 / (1 + 0.001 * (num_linked - 1) ** 2)

This attenuates "broad" entities (those that link to many memories) so
they don't drown out specific matches.

Output: dict mapping ``memory_id -> max_boost_seen``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from litemem.storage.entity_store import EntityStore
from litemem.utils.embeddings import OpenAIEmbedder

logger = logging.getLogger(__name__)

# Identical to ``mem0.utils.scoring.ENTITY_BOOST_WEIGHT``.
ENTITY_BOOST_WEIGHT = 0.5
ENTITY_SIMILARITY_FLOOR = 0.5
MAX_QUERY_ENTITIES = 8


class EntityRetriever:
    def __init__(self, entity_store: EntityStore, embedder: OpenAIEmbedder):
        self.entity_store = entity_store
        self.embedder = embedder

    def compute_boosts(
        self,
        query_entities: List[Tuple[str, str]],
        *,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if not query_entities:
            return {}

        # Dedup (case-insensitive) and cap at MAX_QUERY_ENTITIES.
        seen = set()
        deduped: List[Tuple[str, str]] = []
        for entity_type, entity_text in query_entities[:MAX_QUERY_ENTITIES]:
            key = entity_text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        session_filters = {
            k: v
            for k, v in (filters or {}).items()
            if k in ("user_id", "agent_id", "run_id") and v
        }

        memory_boosts: Dict[str, float] = {}
        for _, entity_text in deduped:
            try:
                vec = self.embedder.embed(entity_text, "search")
                matches = self.entity_store.search(
                    vec, top_k=500, filters=session_filters
                )
            except Exception as e:
                logger.warning(f"Entity boost lookup failed for '{entity_text}': {e}")
                continue

            for match in matches:
                sim = float(match.score or 0.0)
                if sim < ENTITY_SIMILARITY_FLOOR:
                    continue
                linked = (match.payload or {}).get("linked_memory_ids", [])
                if not isinstance(linked, list):
                    continue
                num_linked = max(len(linked), 1)
                memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                boost = sim * ENTITY_BOOST_WEIGHT * memory_count_weight
                for memory_id in linked:
                    if not memory_id:
                        continue
                    mid = str(memory_id)
                    if boost > memory_boosts.get(mid, 0.0):
                        memory_boosts[mid] = boost

        return memory_boosts
