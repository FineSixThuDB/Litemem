"""Entity store — second VexDB-Lite collection used for entity boost retrieval.

Built on top of :class:`storage.vector_store.VexDBVectorStore` by passing a
different table name (``{collection}_entities``). The rows here are entity
records:

    {
      "data":              "Marcus",         # entity surface text
      "entity_type":       "PROPER",         # PROPER / COMPOUND / QUOTED / NOUN
      "linked_memory_ids": ["uuid1", ...],  # memories that mention this entity
      "user_id" / "agent_id" / "run_id":    # session scope (same as parent memory)
    }

The ``linked_memory_ids`` array is what powers the read-pipeline entity boost:
when a query mentions an entity, we vector-search the entity store, then
boost every memory in the matched entity's ``linked_memory_ids`` list.

This file is thin — most logic lives in ``VexDBVectorStore``; we just expose
``search_batch`` (which the write pipeline uses) and serialize/deserialize
the ``linked_memory_ids`` list explicitly because DuckDB stores it inside
the JSON ``payload`` blob.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from litemem.config import VectorStoreConfig
from litemem.data_models import VectorRecord
from litemem.storage.vector_store import VexDBVectorStore

logger = logging.getLogger(__name__)


class EntityStore:
    """Thin wrapper around a second VexDB-Lite collection.

    The wrapper exists for two reasons:
    1. It keeps entity-store responsibilities in their own file.
    2. ``linked_memory_ids`` is a list; serializing it through the JSON
       payload is fine, but we re-cast it to a sorted list of strings on
       read so downstream code can iterate safely.
    """

    def __init__(self, parent_config: VectorStoreConfig):
        entity_config = deepcopy(parent_config)
        entity_config.collection_name = f"{parent_config.collection_name}_entities"
        self.store = VexDBVectorStore(entity_config, table_name=entity_config.collection_name)

    # ------------------------------------------------------------------
    # Payload normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}
        linked = payload.get("linked_memory_ids")
        if isinstance(linked, str):
            try:
                linked = json.loads(linked)
            except Exception:
                linked = []
        if not isinstance(linked, list):
            linked = []
        payload = dict(payload)
        payload["linked_memory_ids"] = [str(x) for x in linked if x]
        return payload

    @classmethod
    def _normalize_record(cls, rec: Optional[VectorRecord]) -> Optional[VectorRecord]:
        if rec is None:
            return None
        rec.payload = cls._normalize_payload(rec.payload)
        return rec

    # ------------------------------------------------------------------
    # CRUD passthroughs
    # ------------------------------------------------------------------

    def insert(self, vectors, ids, payloads):
        self.store.insert(vectors, ids, payloads)

    def update(self, vector_id: str, vector=None, payload=None):
        self.store.update(vector_id, vector=vector, payload=payload)

    def delete(self, vector_id: str) -> None:
        self.store.delete(vector_id)

    def get(self, vector_id: str) -> Optional[VectorRecord]:
        return self._normalize_record(self.store.get(vector_id))

    def search(
        self,
        vectors: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        return [self._normalize_record(r) for r in self.store.search(vectors, top_k, filters) if r]

    def search_batch(
        self,
        vectors_list: List[List[float]],
        top_k: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[VectorRecord]]:
        """No native batch API in DuckDB — fall back to sequential search."""
        return [self.search(v, top_k=top_k, filters=filters) for v in vectors_list]

    def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorRecord]:
        return [self._normalize_record(r) for r in self.store.list(filters, top_k) if r]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.store.reset()

    def close(self) -> None:
        self.store.close()
