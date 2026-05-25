"""Memory extractor — Phase 0-2 of mem0 V3 ``_add_to_vector_store``.

Responsibility:
1. Gather context (last K messages from the SQLite messages buffer + existing
   memories from the vector store).
2. Build the additive-extraction prompt with proper UUID anonymization
   (existing memories get sequential integer IDs to suppress
   ID-hallucination; we keep the mapping so ``linked_memory_ids`` can be
   un-mapped after parsing).
3. Call the LLM with ``response_format={"type": "json_object"}`` and parse
   ``{"memory": [...]}``.

Stage outputs an :class:`ExtractionResult` with the parsed facts, plus the
existing-memory hashes for the deduplicator step.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from litemem.data_models import ExtractedFact, VectorRecord
from litemem.utils.llm_client import OpenAILLM
from litemem.utils.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    generate_additive_extraction_prompt,
)
from litemem.utils.text_utils import extract_json, parse_messages, remove_code_blocks

logger = logging.getLogger(__name__)


_SESSION_DATE_PATTERN = re.compile(
    r"\bSession\s+\d+\s*\([^)]*\bon\s+(\d{1,2}\s+[A-Za-z]+,\s+\d{4})\)",
    flags=re.IGNORECASE,
)
_LOCOMO_TEMPORAL_NOTE = (
    "If New Messages contain session headers like "
    "'Session 1 (1:56 pm on 8 May, 2023)', treat the date in that session "
    "header as the observation date for all following dialogue turns until the "
    "next session header. Resolve relative time phrases such as 'yesterday', "
    "'last Friday', 'last week', and 'recently' against the nearest preceding "
    "session header date, not against today's date and not by copying the "
    "session date unchanged. Write resolved dates as 'D Month YYYY' (for "
    "example, '7 May 2023') rather than YYYY-MM-DD."
)


@dataclass
class ExtractionResult:
    facts: List[ExtractedFact]
    existing_memories: List[VectorRecord]
    existing_hashes: set = field(default_factory=set)
    last_messages: List[Dict[str, Any]] = field(default_factory=list)
    parsed_messages: str = ""


class MemoryExtractor:
    """LLM-driven fact extraction with mem0's ADDITIVE_EXTRACTION_PROMPT."""

    def __init__(self, llm: OpenAILLM, *, custom_instructions: Optional[str] = None):
        self.llm = llm
        self.custom_instructions = custom_instructions

    def extract(
        self,
        messages: List[Dict[str, Any]],
        *,
        existing_memories: List[VectorRecord],
        last_messages: List[Dict[str, Any]],
        filters: Dict[str, Any],
        prompt_override: Optional[str] = None,
    ) -> ExtractionResult:
        """Run Phase 1 retrieval & Phase 2 LLM extraction.

        Args:
            messages: the new chat messages being ingested.
            existing_memories: top-K most-similar memories already in store
                (used both to give the LLM dedup context AND to build the
                ``existing_hashes`` set for hash-based dedup downstream).
            last_messages: the most-recent N messages from the SQLite buffer,
                used to fill the "Last k Messages" section of the prompt.
            filters: the session filters; used only to detect agent-only scope
                so we append :data:`AGENT_CONTEXT_SUFFIX`.
            prompt_override: custom instructions for this call (takes
                precedence over the constructor-level setting).
        """
        parsed_messages = parse_messages(messages)

        # UUID anonymization → suppress LLM hallucination of IDs.
        anonymized = []
        uuid_mapping: Dict[str, str] = {}
        for idx, mem in enumerate(existing_memories):
            uuid_mapping[str(idx)] = mem.id
            anonymized.append({"id": str(idx), "text": mem.payload.get("data", "")})

        existing_hashes = {
            mem.payload.get("hash")
            for mem in existing_memories
            if mem.payload and mem.payload.get("hash")
        }

        # System prompt — optionally add the agent-scope suffix.
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt = system_prompt + AGENT_CONTEXT_SUFFIX

        custom_instr = prompt_override or self.custom_instructions
        observation_date = _extract_observation_date(parsed_messages)
        if _SESSION_DATE_PATTERN.search(parsed_messages):
            custom_instr = _append_instruction(custom_instr, _LOCOMO_TEMPORAL_NOTE)

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=anonymized,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            timestamp=observation_date,
            custom_instructions=custom_instr,
        )

        try:
            raw = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return ExtractionResult(
                facts=[],
                existing_memories=existing_memories,
                existing_hashes=existing_hashes,
                last_messages=last_messages,
                parsed_messages=parsed_messages,
            )

        facts = self._parse_response(raw, uuid_mapping=uuid_mapping)

        return ExtractionResult(
            facts=facts,
            existing_memories=existing_memories,
            existing_hashes=existing_hashes,
            last_messages=last_messages,
            parsed_messages=parsed_messages,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, *, uuid_mapping: Dict[str, str]) -> List[ExtractedFact]:
        """Parse LLM JSON output and resolve anonymized IDs back to real UUIDs."""
        cleaned = remove_code_blocks(raw)
        if not cleaned.strip():
            return []
        try:
            payload = json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            try:
                payload = json.loads(extract_json(cleaned), strict=False)
            except Exception as e:
                logger.error(f"Could not parse extraction response: {e}; raw={raw[:200]!r}")
                return []

        memory_items = MemoryExtractor._coerce_memory_items(payload)

        out: List[ExtractedFact] = []
        for item in memory_items:
            if not isinstance(item, dict):
                continue
            text = (
                item.get("text")
                or item.get("memory")
                or item.get("content")
                or item.get("fact")
            )
            if not text or not isinstance(text, str):
                continue
            # Resolve anonymized linked IDs back to the real UUIDs.
            raw_links = item.get("linked_memory_ids") or []
            linked = []
            if isinstance(raw_links, list):
                for ref in raw_links:
                    ref_str = str(ref)
                    resolved = uuid_mapping.get(ref_str, ref_str)
                    linked.append(resolved)
            out.append(
                ExtractedFact(
                    id=str(item.get("id", "")),
                    text=text,
                    attributed_to=item.get("attributed_to"),
                    linked_memory_ids=linked,
                )
            )
        return out

    @staticmethod
    def _coerce_memory_items(payload: Any) -> List[Any]:
        """Normalize common LLM JSON shapes into a list of memory items.

        The prompt asks for ``{"memory": [...]}``, but OpenAI-compatible
        backends are not all equally strict about JSON-mode object shape. Qwen
        commonly returns a bare JSON array, so accept that without aborting the
        benchmark run.
        """
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            logger.warning(
                "Unexpected extraction response type %s; expected object or list.",
                type(payload).__name__,
            )
            return []

        for key in ("memory", "memories", "facts", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]

        # Some models return one object with the memory text fields directly.
        if any(key in payload for key in ("text", "memory", "content", "fact")):
            return [payload]

        logger.warning(
            "Extraction response object did not contain memory items; keys=%s",
            sorted(str(key) for key in payload.keys())[:20],
        )
        return []


def _extract_observation_date(text: str) -> Optional[str]:
    """Extract the first LoCoMo session date as ISO yyyy-mm-dd when available."""
    match = _SESSION_DATE_PATTERN.search(text or "")
    if not match:
        return None
    raw_date = match.group(1)
    try:
        return datetime.strptime(raw_date, "%d %B, %Y").date().isoformat()
    except ValueError:
        try:
            return datetime.strptime(raw_date, "%d %b, %Y").date().isoformat()
        except ValueError:
            logger.debug("Could not parse observation date from %r", raw_date)
            return None


def _append_instruction(existing: Optional[str], instruction: str) -> str:
    """Append a custom instruction without duplicating it."""
    if not existing:
        return instruction
    if instruction in existing:
        return existing
    return f"{existing}\n\n{instruction}"
