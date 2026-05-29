"""LiteMem — orchestrator class that wires the pipelines together.

Public API matches the methods you'll see on ``mem0.Memory`` (sync only):

    m = LiteMem()
    m.add(messages=[...], user_id="u1")
    m.search("what does the user like?", filters={"user_id": "u1"})
    m.get(memory_id)
    m.get_all(filters={"user_id": "u1"})
    m.update(memory_id, "new text")
    m.delete(memory_id)
    m.delete_all(user_id="u1")
    m.history(memory_id)
    m.reset()

The control flow inside ``add`` and ``search`` mirrors mem0 V3 (see
``write_pipeline/`` and ``read_pipeline/`` for the per-stage docstrings).
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from litemem.config import LiteMemConfig, MemoryConfig  # alias
from litemem.read_pipeline.context_builder import ContextBuilder
from litemem.read_pipeline.entity_retriever import EntityRetriever
from litemem.read_pipeline.keyword_retriever import KeywordRetriever
from litemem.read_pipeline.query_preprocessor import QueryPreprocessor
from litemem.read_pipeline.rank_fusion import RankFusion, normalize_bm25_scores
from litemem.read_pipeline.semantic_retriever import SemanticRetriever
from litemem.storage.entity_store import EntityStore
from litemem.storage.memory_store import MemoryStore, build_session_scope
from litemem.storage.vector_store import VexDBVectorStore
from litemem.utils.embeddings import OpenAIEmbedder
from litemem.utils.llm_client import OpenAILLM
from litemem.utils.text_utils import md5_hash
from litemem.write_pipeline.deduplicator import Deduplicator
from litemem.write_pipeline.entity_linker import EntityLinker
from litemem.write_pipeline.memory_extractor import MemoryExtractor
from litemem.write_pipeline.memory_writer import MemoryWriter
from litemem.write_pipeline.procedural_memory import (
    PROCEDURAL_MEMORY_TYPE,
    ProceduralMemoryCreator,
)

logger = logging.getLogger(__name__)

# Same enum value mem0 exposes as ``MemoryType.PROCEDURAL.value``.
PROCEDURAL_MEMORY = PROCEDURAL_MEMORY_TYPE


# ---------------------------------------------------------------------------
# Helpers (mirroring mem0's _build_filters_and_metadata)
# ---------------------------------------------------------------------------

def _validate_entity_id(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"Invalid {name}: empty or whitespace-only.")
    if any(c.isspace() for c in trimmed):
        raise ValueError(f"Invalid {name}: cannot contain whitespace.")
    return trimmed


def _build_filters_and_metadata(
    *,
    user_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
):
    base_metadata = deepcopy(input_metadata) if input_metadata else {}
    effective_filters = deepcopy(input_filters) if input_filters else {}

    user_id = _validate_entity_id(user_id, "user_id")
    agent_id = _validate_entity_id(agent_id, "agent_id")
    run_id = _validate_entity_id(run_id, "run_id")

    provided = []
    if user_id:
        base_metadata["user_id"] = user_id
        effective_filters["user_id"] = user_id
        provided.append("user_id")
    if agent_id:
        base_metadata["agent_id"] = agent_id
        effective_filters["agent_id"] = agent_id
        provided.append("agent_id")
    if run_id:
        base_metadata["run_id"] = run_id
        effective_filters["run_id"] = run_id
        provided.append("run_id")

    if not provided:
        raise ValueError(
            "At least one of user_id / agent_id / run_id must be provided."
        )
    return base_metadata, effective_filters


def _session_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in (filters or {}).items()
        if k in ("user_id", "agent_id", "run_id") and v
    }


def _normalize_messages(messages: Union[str, dict, List[dict]]) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        return [messages]
    if isinstance(messages, list):
        return messages
    raise ValueError("messages must be str, dict, or list[dict]")


# ---------------------------------------------------------------------------
# LiteMem
# ---------------------------------------------------------------------------

class LiteMem:
    """LiteMem — the public memory system class."""

    def __init__(self, config: Optional[Union[LiteMemConfig, MemoryConfig]] = None):
        self.config = config or LiteMemConfig()

        # --- I/O layers ---
        self.llm = OpenAILLM(self.config.llm)
        self.embedder = OpenAIEmbedder(self.config.embedder)
        self.llm.usage_callback = self.config.usage_callback
        self.embedder.usage_callback = self.config.usage_callback
        self.technique_flags = self.config.technique_flags
        self.vector_store = VexDBVectorStore(self.config.vector_store)
        self.memory_store = MemoryStore(
            self.config.history_db_path,
            recent_messages_limit=self.config.recent_messages_limit,
        )
        # Entity store is lazy — many callers will never need it.
        self._entity_store: Optional[EntityStore] = None

        # --- write pipeline stages ---
        self.extractor = MemoryExtractor(
            self.llm, custom_instructions=self.config.custom_instructions
        )
        self.deduplicator = Deduplicator()
        self.writer = MemoryWriter(self.vector_store, self.memory_store, self.embedder)
        self._procedural = ProceduralMemoryCreator(self.llm, self.writer)
        self._entity_linker: Optional[EntityLinker] = None

        # --- read pipeline stages ---
        self.query_preprocessor = QueryPreprocessor()
        self.semantic_retriever = SemanticRetriever(self.vector_store, self.embedder)
        self.keyword_retriever = KeywordRetriever(self.vector_store)
        self.context_builder = ContextBuilder()
        # entity_retriever needs the entity store; lazy-init like in mem0.
        self._entity_retriever: Optional[EntityRetriever] = None

        self.rank_fusion = RankFusion()

    def _emit_stage_event(self, stage: str, latency_s: float, **extra: Any) -> None:
        callback = getattr(self.config, "usage_callback", None)
        if callback is None:
            return
        event = {
            "stage": stage,
            "kind": "local",
            "latency_s": latency_s,
            "chat_input_tokens": 0,
            "chat_output_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "embedding_tokens": 0,
            "total_tokens": 0,
        }
        if extra:
            event["extra"] = extra
        callback(event)

    # ------------------------------------------------------------------
    # Lazy entity infra
    # ------------------------------------------------------------------

    @property
    def entity_store(self) -> EntityStore:
        if self._entity_store is None:
            self._entity_store = EntityStore(self.config.vector_store)
        return self._entity_store

    @property
    def entity_linker(self) -> EntityLinker:
        if self._entity_linker is None:
            self._entity_linker = EntityLinker(self.entity_store, self.embedder)
        return self._entity_linker

    @property
    def entity_retriever(self) -> EntityRetriever:
        if self._entity_retriever is None:
            self._entity_retriever = EntityRetriever(self.entity_store, self.embedder)
        return self._entity_retriever

    # ==================================================================
    # ADD
    # ==================================================================

    def add(
        self,
        messages: Union[str, dict, List[dict]],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store new memories. Returns ``{"results": [...]}``."""
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != PROCEDURAL_MEMORY:
            raise ValueError(
                f"Invalid memory_type. Pass {PROCEDURAL_MEMORY!r} for procedural memories."
            )

        messages = _normalize_messages(messages)

        # Procedural-memory branch: agent_id is required by mem0 too.
        if memory_type == PROCEDURAL_MEMORY and agent_id is not None:
            return self._procedural.create(
                messages, metadata=processed_metadata, prompt=prompt
            )

        # infer=False / ablated extraction branch — store each message verbatim, no LLM.
        if not infer or not self.technique_flags.use_additive_extraction:
            results = []
            for msg in messages:
                if not isinstance(msg, dict) or not msg.get("role") or not msg.get("content"):
                    continue
                if msg["role"] == "system":
                    continue
                per_msg_meta = deepcopy(processed_metadata)
                per_msg_meta["role"] = msg["role"]
                actor_name = msg.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name
                mem_id = self.writer.write_raw_message(msg["content"], metadata=per_msg_meta)
                results.append(
                    {
                        "id": mem_id,
                        "memory": msg["content"],
                        "event": "ADD",
                        "actor_id": actor_name,
                        "role": msg["role"],
                    }
                )
            return {"results": results}

        # === V3 phased batch pipeline ===
        session_scope = build_session_scope(effective_filters)
        if self.technique_flags.use_recent_messages_context:
            last_messages = self.memory_store.get_last_messages(
                session_scope, limit=self.config.recent_messages_limit
            )
        else:
            last_messages = []

        session_only_filters = _session_filters(effective_filters)
        # Phase 1: existing memory retrieval (over the session-scope only).
        existing_results = []
        if self.technique_flags.use_existing_memory_context:
            query_text = "\n".join(
                f"{m.get('role','')}: {m.get('content','')}" for m in messages
            )
            try:
                stage_start = time.perf_counter()
                query_vec = self.embedder.embed(
                    query_text,
                    "search",
                    usage_stage="add.existing_memory_lookup.embedding",
                )
                existing_results = self.vector_store.search(
                    query_vec,
                    top_k=self.config.existing_memories_limit,
                    filters=session_only_filters,
                )
                self._emit_stage_event(
                    "add.existing_memory_lookup.vector_search",
                    time.perf_counter() - stage_start,
                    result_count=len(existing_results),
                )
            except Exception as e:
                logger.warning(f"Existing-memory retrieval failed: {e}")
                existing_results = []

        # Phase 2: LLM extraction.
        extraction = self.extractor.extract(
            messages,
            existing_memories=existing_results,
            last_messages=last_messages,
            filters=effective_filters,
            prompt_override=prompt,
            use_uuid_anonymization=self.technique_flags.use_uuid_anonymization,
            use_json_response_format=self.technique_flags.use_json_response_format,
        )

        # Phases 4-5: hash dedup.
        if self.technique_flags.use_hash_dedup:
            kept_facts, kept_hashes = self.deduplicator.filter(
                extraction.facts, existing_hashes=extraction.existing_hashes
            )
        else:
            kept_facts = list(extraction.facts)
            kept_hashes = [md5_hash(f.text) for f in kept_facts]

        # Phases 3 + 6 + 8: batch embed + persist + save messages.
        written = self.writer.write_facts(
            kept_facts,
            kept_hashes,
            metadata=processed_metadata,
            messages=messages,
            session_scope=session_scope,
        )

        # Phase 7: batch entity linking (best-effort, swallow failures).
        if written and self.technique_flags.use_entity_boost:
            try:
                self.entity_linker.link_memories(written, filters=effective_filters)
            except Exception as e:
                logger.warning(f"Entity linking failed: {e}")

        return {
            "results": [
                {"id": mid, "memory": text, "event": "ADD"} for mid, text in written
            ]
        }

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Hybrid search (semantic + BM25 + entity boost)."""
        effective_filters = dict(filters or {})
        for k in ("user_id", "agent_id", "run_id"):
            if k in effective_filters:
                effective_filters[k] = _validate_entity_id(effective_filters[k], k)
        if not any(k in effective_filters for k in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of user_id / agent_id / run_id."
            )
        session_only = _session_filters(effective_filters)

        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative int.")
        if threshold is None:
            threshold = 0.1
        elif not (0 <= threshold <= 1):
            raise ValueError("threshold must be between 0 and 1.")

        # Step 1: preprocess (lemmatize + entities).
        pre = self.query_preprocessor.preprocess(query)

        # Step 3: semantic search (over-fetched).
        semantic_results = self.semantic_retriever.retrieve(
            query, top_k=top_k, filters=effective_filters
        )

        # Step 4: keyword search via rank_bm25 over session corpus.
        raw_bm25 = []
        if self.technique_flags.use_bm25:
            stage_start = time.perf_counter()
            raw_bm25 = self.keyword_retriever.retrieve(
                pre.lemmatized,
                filters=session_only,
                top_k=max(top_k * 4, 60),
            )
            self._emit_stage_event(
                "search.bm25",
                time.perf_counter() - stage_start,
                result_count=len(raw_bm25),
            )
        # Step 5: BM25 sigmoid normalization.
        bm25_scores = normalize_bm25_scores(raw_bm25, query_num_terms=pre.num_terms)

        # Step 6: entity boost.
        entity_boosts: Dict[str, float] = {}
        if pre.entities and self.technique_flags.use_entity_boost:
            try:
                entity_boosts = self.entity_retriever.compute_boosts(
                    pre.entities, filters=session_only
                )
            except Exception as e:
                logger.warning(f"Entity boost computation failed: {e}")

        # Steps 7-8: additive fusion + top-K.
        scored = self.rank_fusion.fuse(
            semantic_results,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=top_k,
        )

        # Step 9: format.
        return {"results": self.context_builder.format_scored(scored)}

    # ==================================================================
    # CRUD
    # ==================================================================

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        record = self.vector_store.get(memory_id)
        if record is None:
            return None
        return self.context_builder.format_record(record)

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        effective_filters = dict(filters or {})
        for k in ("user_id", "agent_id", "run_id"):
            if k in effective_filters:
                effective_filters[k] = _validate_entity_id(effective_filters[k], k)
        if not any(k in effective_filters for k in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of user_id / agent_id / run_id."
            )
        records = self.vector_store.list(filters=effective_filters, top_k=top_k)
        return {"results": self.context_builder.format_list(records)}

    def update(
        self,
        memory_id: str,
        data: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mid, new_payload = self.writer.update_memory(memory_id, data, metadata=metadata)
        session_filters = _session_filters(new_payload)
        try:
            self.entity_linker.remove_memory_from_entities(mid, filters=session_filters)
            self.entity_linker.link_single_memory(mid, data, filters=session_filters)
        except Exception as e:
            logger.debug(f"Entity refresh on update failed: {e}")
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: str) -> Dict[str, Any]:
        payload = self.writer.delete_memory(memory_id)
        session_filters = _session_filters(payload)
        try:
            self.entity_linker.remove_memory_from_entities(memory_id, filters=session_filters)
        except Exception as e:
            logger.debug(f"Entity cleanup on delete failed: {e}")
        return {"message": "Memory deleted successfully!"}

    def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id
        if not filters:
            raise ValueError(
                "delete_all requires at least one of user_id / agent_id / run_id. "
                "Use reset() to wipe the entire store."
            )
        records = self.vector_store.list(filters=filters)
        for rec in records:
            try:
                self.delete(rec.id)
            except Exception as e:
                logger.warning(f"Failed to delete memory {rec.id}: {e}")
        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        return self.memory_store.get_history(memory_id)

    # ==================================================================
    # Reset / teardown
    # ==================================================================

    def reset(self) -> None:
        """Drop all memory data and rebuild the stores."""
        self.memory_store.reset()
        self.vector_store.reset()
        if self._entity_store is not None:
            try:
                self._entity_store.reset()
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None
            self._entity_linker = None
            self._entity_retriever = None

    def close(self) -> None:
        try:
            self.memory_store.close()
        except Exception:
            pass
        try:
            self.vector_store.close()
        except Exception:
            pass
        if self._entity_store is not None:
            try:
                self._entity_store.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
