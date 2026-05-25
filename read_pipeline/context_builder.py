"""Context builder — Step 9 of mem0 V3 search.

Turn the ranked :class:`ScoredMemory` list into a list of plain dicts that
match the shape mem0 returns (with promoted session fields at the top
level and the rest tucked under ``metadata``).
"""

from __future__ import annotations

from typing import Any, Dict, List

from litemem.data_models import (
    CORE_PAYLOAD_KEYS,
    MemoryItem,
    PROMOTED_PAYLOAD_KEYS,
    ScoredMemory,
    VectorRecord,
)


class ContextBuilder:
    @staticmethod
    def format_scored(scored: List[ScoredMemory]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in scored:
            payload = s.payload or {}
            if not payload.get("data"):
                # Skip rows where the data column was somehow lost.
                continue
            item = MemoryItem(
                id=s.id,
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=s.score,
            )
            for key in PROMOTED_PAYLOAD_KEYS:
                v = payload.get(key)
                if v is not None:
                    setattr(item, key, v)
            additional = {
                k: v for k, v in payload.items()
                if k not in CORE_PAYLOAD_KEYS
            }
            if additional:
                item.metadata = additional
            out.append(item.to_dict())
        return out

    @staticmethod
    def format_record(record: VectorRecord) -> Dict[str, Any]:
        payload = record.payload or {}
        item = MemoryItem(
            id=record.id,
            memory=payload.get("data", ""),
            hash=payload.get("hash"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            score=record.score,
        )
        for key in PROMOTED_PAYLOAD_KEYS:
            v = payload.get(key)
            if v is not None:
                setattr(item, key, v)
        additional = {
            k: v for k, v in payload.items()
            if k not in CORE_PAYLOAD_KEYS
        }
        if additional:
            item.metadata = additional
        return item.to_dict()

    @classmethod
    def format_list(cls, records: List[VectorRecord]) -> List[Dict[str, Any]]:
        return [cls.format_record(r) for r in records if r is not None]
