"""Procedural memory creator — agent-only write path.

Procedural memories are LLM-generated summaries of an agent's recent
execution history (action / observation / next step). They are stored in
the same vector collection as ordinary memories, but tagged with
``memory_type = "procedural_memory"`` in the payload.

Ported from mem0/memory/main.py ``_create_procedural_memory``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from litemem.utils.llm_client import OpenAILLM
from litemem.utils.prompts import PROCEDURAL_MEMORY_SYSTEM_PROMPT
from litemem.utils.text_utils import remove_code_blocks
from litemem.write_pipeline.memory_writer import MemoryWriter

logger = logging.getLogger(__name__)

PROCEDURAL_MEMORY_TYPE = "procedural_memory"


class ProceduralMemoryCreator:
    def __init__(self, llm: OpenAILLM, writer: MemoryWriter):
        self.llm = llm
        self.writer = writer

    def create(
        self,
        messages: List[Dict[str, Any]],
        *,
        metadata: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if metadata is None:
            raise ValueError("Metadata is required for procedural memory.")

        parsed = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]
        try:
            raw = self.llm.generate_response(
                messages=parsed,
                usage_stage="add.procedural_memory_generation",
            )
        except Exception as e:
            logger.error(f"Procedural memory generation failed: {e}")
            raise
        procedural_text = remove_code_blocks(raw)

        full_metadata = dict(metadata)
        full_metadata["memory_type"] = PROCEDURAL_MEMORY_TYPE

        memory_id = self.writer.write_raw_message(procedural_text, metadata=full_metadata)
        return {
            "results": [
                {"id": memory_id, "memory": procedural_text, "event": "ADD"}
            ]
        }
