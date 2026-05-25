"""Rank fusion — Steps 5, 7-8 of mem0 V3 search.

Combines three signals into one score, exactly mirroring mem0:

    semantic_score   ∈ [0, 1]   from the vector store similarity
    bm25_score       ∈ [0, 1]   sigmoid-normalized raw BM25 (query-length adaptive)
    entity_boost     ∈ [0, 0.5] from EntityRetriever

The combined score uses an *adaptive divisor* so partial signal sets don't
unfairly penalize candidates:

    max_possible = 1.0
    if bm25_used:   max_possible += 1.0
    if entity_used: max_possible += 0.5
    combined = (semantic + bm25 + entity) / max_possible

The semantic-only threshold is applied BEFORE combining, so candidates
below the threshold are dropped even if BM25/entity would have raised them.
This is the same gating mem0 uses.
"""

from __future__ import annotations

import math
from typing import Dict, List

from litemem.data_models import ScoredMemory, VectorRecord

ENTITY_BOOST_WEIGHT = 0.5  # must match ``entity_retriever.ENTITY_BOOST_WEIGHT``


# ---------------------------------------------------------------------------
# Sigmoid BM25 normalization
# ---------------------------------------------------------------------------

def get_bm25_params(num_terms: int) -> tuple:
    """Query-length-adaptive sigmoid parameters (midpoint, steepness)."""
    if num_terms <= 3:
        return 5.0, 0.7
    if num_terms <= 6:
        return 7.0, 0.6
    if num_terms <= 9:
        return 9.0, 0.5
    if num_terms <= 15:
        return 10.0, 0.5
    return 12.0, 0.5


def normalize_bm25(raw: float, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (raw - midpoint)))


def normalize_bm25_scores(
    raw_pairs: List[tuple],
    *,
    query_num_terms: int,
) -> Dict[str, float]:
    if not raw_pairs:
        return {}
    midpoint, steepness = get_bm25_params(query_num_terms)
    return {
        mid: normalize_bm25(raw, midpoint, steepness)
        for mid, raw in raw_pairs
        if raw > 0
    }


# ---------------------------------------------------------------------------
# Additive fusion
# ---------------------------------------------------------------------------

class RankFusion:
    @staticmethod
    def fuse(
        semantic_results: List[VectorRecord],
        *,
        bm25_scores: Dict[str, float],
        entity_boosts: Dict[str, float],
        threshold: float,
        top_k: int,
    ) -> List[ScoredMemory]:
        has_bm25 = bool(bm25_scores)
        has_entity = bool(entity_boosts)

        max_possible = 1.0
        if has_bm25:
            max_possible += 1.0
        if has_entity:
            max_possible += ENTITY_BOOST_WEIGHT

        scored: List[ScoredMemory] = []
        for rec in semantic_results:
            sem_score = float(rec.score or 0.0)
            if sem_score < threshold:
                continue
            mid = str(rec.id)
            combined = (
                sem_score
                + bm25_scores.get(mid, 0.0)
                + entity_boosts.get(mid, 0.0)
            )
            scored.append(
                ScoredMemory(
                    id=mid,
                    score=min(combined / max_possible, 1.0),
                    payload=rec.payload or {},
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
