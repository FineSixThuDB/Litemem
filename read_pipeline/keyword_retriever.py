"""Keyword retriever — BM25 over a session-scoped corpus.

VexDB-Lite has no native BM25. Per the user's chosen strategy we build the
index on demand at query time:

1. Pull every memory's ``(id, text_lemmatized)`` for the current session
   scope from the vector store (one cheap projection — only promoted
   columns, no JSON payload parsing).
2. Build a ``rank_bm25.BM25Okapi`` index over those lemma strings.
3. Score the (lemmatized) query and return ``[(id, raw_bm25_score)]`` for
   anything with score > 0.

Sigmoid normalization of the raw scores happens later in
:mod:`rank_fusion`, because it depends on the query length and we want
to keep the retriever's output decoupled.

Notes:
- If ``rank_bm25`` is not installed, this retriever returns an empty list
  and ``rank_fusion`` will simply degrade to "semantic + entity boost".
- For very large session corpora (>100k memories) building the index per
  call gets slow; in that case switch to "process-level persistent index".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from litemem.storage.vector_store import VexDBVectorStore

logger = logging.getLogger(__name__)


class KeywordRetriever:
    def __init__(self, vector_store: VexDBVectorStore, *, corpus_cap: int = 50000):
        self.vector_store = vector_store
        self.corpus_cap = corpus_cap
        self._bm25_available = self._try_import()

    @staticmethod
    def _try_import() -> bool:
        try:
            import rank_bm25  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "rank_bm25 not installed; keyword retrieval disabled. "
                "Install with: pip install rank_bm25"
            )
            return False

    def retrieve(
        self,
        query_lemmatized: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 60,
    ) -> List[Tuple[str, float]]:
        """Return ``[(memory_id, raw_bm25_score), ...]`` for non-zero matches.

        ``top_k`` is the over-fetch limit; final ranking happens later.
        Empty list when BM25 is disabled, query has no tokens, or the
        session corpus is empty.
        """
        if not self._bm25_available or not query_lemmatized.strip():
            return []

        from rank_bm25 import BM25Okapi

        rows = self.vector_store.list_pairs(
            filters=filters,
            top_k=self.corpus_cap,
            fields=("id", "text_lemmatized"),
        )
        if not rows:
            return []

        ids: List[str] = []
        tokenized_corpus: List[List[str]] = []
        for row in rows:
            mid, lemma = row[0], row[1] or ""
            if not lemma:
                continue
            tokens = lemma.split()
            if not tokens:
                continue
            ids.append(str(mid))
            tokenized_corpus.append(tokens)

        if not tokenized_corpus:
            return []

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query_lemmatized.split())
        except Exception as e:
            logger.warning(f"BM25 scoring failed: {e}")
            return []

        pairs = [
            (ids[i], float(scores[i]))
            for i in range(len(ids))
            if scores[i] > 0.0
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]
