"""Deduplicator — Phase 4-5 of mem0 V3 add().

The deduplicator removes:
1. Facts whose md5(text) hash matches an existing memory (cross-batch dedup).
2. Facts whose md5(text) hash matches a fact already in the same batch
   (intra-batch dedup).

We use md5 of the raw text exactly like mem0:

    mem_hash = hashlib.md5(text.encode()).hexdigest()

mem0 stores this hash on every memory's payload so the next add() can build
``existing_hashes`` cheaply. LiteMem preserves the same field name and
semantics.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Set, Tuple

from litemem.data_models import ExtractedFact
from litemem.utils.text_utils import md5_hash

logger = logging.getLogger(__name__)


class Deduplicator:
    """Hash-based dedup, both against existing store and within-batch."""

    @staticmethod
    def filter(
        facts: List[ExtractedFact],
        *,
        existing_hashes: Iterable[str],
    ) -> Tuple[List[ExtractedFact], List[str]]:
        """Return ``(kept_facts, hashes_of_kept_facts)``.

        Order of ``kept_facts`` matches the input order (so the LLM's chosen
        id sequence is preserved upstream).
        """
        existing: Set[str] = set(h for h in existing_hashes if h)
        seen: Set[str] = set()
        kept: List[ExtractedFact] = []
        hashes: List[str] = []
        for fact in facts:
            h = md5_hash(fact.text)
            if h in existing:
                logger.debug(f"Skipping duplicate (matches store): {fact.text[:50]}")
                continue
            if h in seen:
                logger.debug(f"Skipping duplicate (within batch):  {fact.text[:50]}")
                continue
            seen.add(h)
            kept.append(fact)
            hashes.append(h)
        return kept, hashes
