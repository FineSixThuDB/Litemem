"""LOCOMO-style evaluation metrics.

This is a placeholder so the directory has a working baseline; the heavy
lifting for full LOCOMO evaluation lives in ``MemoryDataBenchmark/``
(outside this package). For now we expose:

- ``token_count``  : rough token estimate (OpenAI tiktoken if available)
- ``recall_at_k``  : recall@k against a ground-truth set
- ``mrr``          : mean reciprocal rank
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Set

logger = logging.getLogger(__name__)


def token_count(text: str, *, model: str = "gpt-4o-mini") -> int:
    """Best-effort token counter. Uses tiktoken when available; falls back
    to ``len(text.split())`` so a missing dep doesn't break evaluation."""
    try:
        import tiktoken
    except ImportError:
        return len(text.split())
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def recall_at_k(predicted_ids: List[str], gold_ids: Iterable[str], k: int) -> float:
    gold: Set[str] = set(gold_ids)
    if not gold:
        return 0.0
    topk = set(predicted_ids[:k])
    return len(topk & gold) / len(gold)


def mrr(predicted_ids: List[str], gold_ids: Iterable[str]) -> float:
    gold: Set[str] = set(gold_ids)
    if not gold:
        return 0.0
    for rank, pid in enumerate(predicted_ids, start=1):
        if pid in gold:
            return 1.0 / rank
    return 0.0
