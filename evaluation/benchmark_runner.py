"""Minimal benchmark runner skeleton — LOCOMO-style.

This is intentionally simple: it loads a JSONL benchmark file with the
schema below, runs ``LiteMem.add`` and ``LiteMem.search`` for each example,
and reports aggregate metrics + LLM-token estimate.

Expected JSONL schema (one example per line):

    {
      "user_id": "u123",
      "ingest": [{"role": "user", "content": "..."}, ...],
      "queries": [
        {"query": "...", "gold_memory_ids": ["..."], "k": 5}
      ]
    }

Full LOCOMO evaluation (with per-conversation accuracy judging by an
external LLM) belongs in ``MemoryDataBenchmark/``; this runner is for
quick local smoke testing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litemem.config import LiteMemConfig
from litemem.evaluation.metrics import mrr, recall_at_k, token_count
from litemem.main import LiteMem

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    num_examples: int = 0
    num_queries: int = 0
    sum_recall_k: float = 0.0
    sum_mrr: float = 0.0
    total_add_seconds: float = 0.0
    total_search_seconds: float = 0.0
    total_input_tokens: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_recall(self) -> float:
        return self.sum_recall_k / self.num_queries if self.num_queries else 0.0

    @property
    def avg_mrr(self) -> float:
        return self.sum_mrr / self.num_queries if self.num_queries else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "num_queries": self.num_queries,
            "avg_recall_at_k": self.avg_recall,
            "avg_mrr": self.avg_mrr,
            "total_add_seconds": self.total_add_seconds,
            "total_search_seconds": self.total_search_seconds,
            "estimated_input_tokens": self.total_input_tokens,
            **self.extras,
        }


def run_benchmark(jsonl_path: str, config: Optional[LiteMemConfig] = None) -> BenchmarkResult:
    """Run a tiny benchmark over a JSONL file. Resets the store between
    examples so cross-example bleed-through doesn't corrupt recall."""
    config = config or LiteMemConfig()
    mem = LiteMem(config)
    result = BenchmarkResult()

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                result.num_examples += 1

                mem.reset()

                user_id = ex.get("user_id", "default")
                ingest_messages: List[Dict[str, Any]] = ex.get("ingest", [])

                # Token estimate for the inputs we'd send to the LLM.
                joined = "\n".join(m.get("content", "") for m in ingest_messages)
                result.total_input_tokens += token_count(joined, model=config.llm.model)

                t0 = time.time()
                if ingest_messages:
                    mem.add(ingest_messages, user_id=user_id)
                result.total_add_seconds += time.time() - t0

                for q in ex.get("queries", []):
                    query = q.get("query")
                    gold = q.get("gold_memory_ids", [])
                    k = q.get("k", 5)
                    if not query:
                        continue
                    t0 = time.time()
                    out = mem.search(
                        query, top_k=k, filters={"user_id": user_id}
                    )
                    result.total_search_seconds += time.time() - t0
                    predicted = [r["id"] for r in out.get("results", [])]
                    result.num_queries += 1
                    result.sum_recall_k += recall_at_k(predicted, gold, k)
                    result.sum_mrr += mrr(predicted, gold)
    finally:
        mem.close()

    return result
