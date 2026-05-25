"""Query preprocessor — Step 1 of mem0 V3 ``_search_vector_store``.

Two cheap, CPU-only steps:

1. ``lemmatize_for_bm25`` over the query — produces the same lemma form
   we stored in ``text_lemmatized`` at write time so BM25 matches stay
   comparable.
2. ``extract_entities`` over the query — used by the entity retriever
   to compute the entity boost.

We also memo the number of lemma tokens because the BM25 sigmoid
parameters depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from litemem.utils.text_utils import extract_entities, lemmatize_for_bm25


@dataclass
class PreprocessedQuery:
    raw: str
    lemmatized: str
    entities: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def num_terms(self) -> int:
        return len(self.lemmatized.split()) if self.lemmatized else 1


class QueryPreprocessor:
    @staticmethod
    def preprocess(query: str) -> PreprocessedQuery:
        lemma = lemmatize_for_bm25(query or "")
        try:
            entities = extract_entities(query or "")
        except Exception:
            entities = []
        return PreprocessedQuery(raw=query or "", lemmatized=lemma, entities=entities)
