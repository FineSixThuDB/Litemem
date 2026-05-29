"""Semantic retriever — Step 3 of mem0 V3 search.

Just a thin wrapper that:
1. Embeds the query via the configured embedder.
2. Calls ``vector_store.search`` with the over-fetch limit
   (max(top_k * 4, 60), per mem0).

Output is a list of :class:`VectorRecord` (id / similarity score / payload).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from litemem.data_models import VectorRecord
from litemem.storage.vector_store import VexDBVectorStore
from litemem.utils.embeddings import OpenAIEmbedder


class SemanticRetriever:
    def __init__(self, vector_store: VexDBVectorStore, embedder: OpenAIEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        internal_limit = max(top_k * 4, 60)
        vec = self.embedder.embed(
            query,
            "search",
            usage_stage="search.semantic_embedding",
        )
        return self.vector_store.search(vec, top_k=internal_limit, filters=filters)

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed(
            query,
            "search",
            usage_stage="search.semantic_embedding",
        )
