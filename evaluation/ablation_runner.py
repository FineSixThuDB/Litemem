"""Token-efficient ablation runner for LiteMem.

The runner keeps attribution simple:
- run one full configuration and leave-one-out ablations;
- preserve per-operation usage events;
- summarize costs per query, per config, and per workload group.

It can read a small JSONL schema directly or reuse MemoryDataBenchmark's
``ConversationCreator`` when agent/dataset YAML files are supplied.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from litemem.config import (
    EmbedderConfig,
    LiteMemConfig,
    LLMConfig,
    TechniqueFlags,
    VectorStoreConfig,
)
from litemem.evaluation.metrics import mrr, recall_at_k, token_count
from litemem.main import LiteMem
from litemem.utils.llm_client import OpenAILLM


CONFIG_TECHNIQUES = {
    "C_FULL": "none",
    "C_MINUS_L2_EXISTING_CONTEXT": "L2_existing_memory_context",
    "C_MINUS_L3_RECENT_MESSAGES": "L3_recent_messages_context",
    "C_MINUS_L4_UUID_ANON": "L4_uuid_anonymization",
    "C_MINUS_L5_JSON_RESPONSE_FORMAT": "L5_json_response_format",
    "C_MINUS_R2_BM25_RERANK": "R2_bm25_rerank",
    "C_MINUS_R3_ENTITY_BOOST": "R3_entity_linking_boost",
    "C_MINUS_P1_HASH_DEDUP": "P1_hash_dedup",
    "C_RAW_STORE_NO_L1": "L1_additive_extraction_diagnostic",
}


@dataclass
class AblationExample:
    context_id: str
    user_id: str
    ingest: List[Dict[str, Any]]
    queries: List[Dict[str, Any]]
    eval_metadata: Dict[str, Any] = field(default_factory=dict)


class UsageLedger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}

    def set_context(
        self,
        *,
        config: str,
        changed_technique: str,
        context_id: str,
        query_id: Optional[str] = None,
    ) -> None:
        self.context = {
            "run_id": self.run_id,
            "config": config,
            "changed_technique": changed_technique,
            "context_id": context_id,
            "query_id": query_id,
        }

    def callback(self, event: Dict[str, Any]) -> None:
        merged = dict(self.context)
        merged.update(event or {})
        stage = str(merged.get("stage") or "")
        merged.setdefault("operation", stage.split(".", 1)[0] if "." in stage else stage)
        for key in (
            "chat_input_tokens",
            "chat_output_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "embedding_tokens",
            "total_tokens",
        ):
            merged[key] = int(merged.get(key) or 0)
        merged["latency_s"] = float(merged.get("latency_s") or 0.0)
        merged["ts"] = time.time()
        self.events.append(merged)

    def write_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")


def technique_flags_for_config(config_id: str) -> TechniqueFlags:
    flags = TechniqueFlags()
    if config_id == "C_MINUS_L2_EXISTING_CONTEXT":
        flags.use_existing_memory_context = False
    elif config_id == "C_MINUS_L3_RECENT_MESSAGES":
        flags.use_recent_messages_context = False
    elif config_id == "C_MINUS_L4_UUID_ANON":
        flags.use_uuid_anonymization = False
    elif config_id == "C_MINUS_L5_JSON_RESPONSE_FORMAT":
        flags.use_json_response_format = False
    elif config_id == "C_MINUS_R2_BM25_RERANK":
        flags.use_bm25 = False
    elif config_id == "C_MINUS_R3_ENTITY_BOOST":
        flags.use_entity_boost = False
    elif config_id == "C_MINUS_P1_HASH_DEDUP":
        flags.use_hash_dedup = False
    elif config_id == "C_RAW_STORE_NO_L1":
        flags.use_additive_extraction = False
    return flags


def iter_ablation_configs(include_raw_diagnostic: bool = True) -> List[Tuple[str, str, TechniqueFlags]]:
    ids = list(CONFIG_TECHNIQUES.keys())
    if not include_raw_diagnostic:
        ids.remove("C_RAW_STORE_NO_L1")
    return [
        (config_id, CONFIG_TECHNIQUES[config_id], technique_flags_for_config(config_id))
        for config_id in ids
    ]


def _flatten_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return [str(value)]


def _source_ids_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ("locomo_source_ids", "source_ids", "source_id", "target_source_ids"):
        ids.extend(_flatten_strings(metadata.get(key)))
    return _dedupe(ids)


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def normalize_ingest_item(item: Any, index: int) -> Dict[str, Any]:
    if isinstance(item, str):
        return {"content": item, "metadata": {"source_id": f"chunk_{index}"}}
    if not isinstance(item, dict):
        return {"content": str(item), "metadata": {"source_id": f"chunk_{index}"}}

    content = item.get("content") or item.get("text") or item.get("page_content") or ""
    metadata = dict(item.get("metadata") or {})
    for key in ("source_id", "source_ids", "locomo_source_ids", "chunk_id"):
        if key in item and key not in metadata:
            metadata[key] = item[key]
    if not _source_ids_from_metadata(metadata):
        metadata["source_id"] = str(item.get("id") or item.get("chunk_id") or f"chunk_{index}")
    return {"content": str(content), "metadata": metadata}


def normalize_query_item(item: Any, index: int) -> Dict[str, Any]:
    if isinstance(item, str):
        return {"query_id": f"q_{index}", "query": item}
    if not isinstance(item, dict):
        return {"query_id": f"q_{index}", "query": str(item)}
    out = dict(item)
    out.setdefault("query_id", out.get("id") or out.get("qa_pair_id") or f"q_{index}")
    out.setdefault("query", out.get("question") or out.get("input") or "")
    return out


def load_jsonl_examples(path: str) -> List[AblationExample]:
    examples: List[AblationExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            context_id = str(raw.get("context_id") or f"context_{idx}")
            user_id = str(raw.get("user_id") or context_id)
            ingest = [
                normalize_ingest_item(item, i)
                for i, item in enumerate(raw.get("ingest") or raw.get("context") or [])
            ]
            queries = [
                normalize_query_item(item, i)
                for i, item in enumerate(raw.get("queries") or [])
            ]
            examples.append(
                AblationExample(
                    context_id=context_id,
                    user_id=user_id,
                    ingest=ingest,
                    queries=queries,
                    eval_metadata=dict(raw.get("eval_metadata") or {}),
                )
            )
    return examples


def _memory_data_benchmark_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MemoryDataBenchmark"))


def _ensure_memory_data_benchmark_on_path() -> str:
    bench_root = _memory_data_benchmark_root()
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)
    return bench_root


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_memory_data_benchmark_answer_prompts(
    *,
    agent_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    _ensure_memory_data_benchmark_on_path()
    from benchmark.memoryagentbench.prompts.benchmark_templates import get_template

    sub_dataset = dataset_config.get("sub_dataset")
    agent_name = agent_config.get("agent_name")
    if not sub_dataset or not agent_name:
        return None, None
    return (
        get_template(sub_dataset, "memory_answer", agent_name),
        get_template(sub_dataset, "system", agent_name),
    )


def load_memory_data_benchmark_examples(
    *,
    agent_config_path: str,
    dataset_config_path: str,
    max_contexts: Optional[int] = None,
    max_queries_per_context: Optional[int] = None,
) -> List[AblationExample]:
    bench_root = _ensure_memory_data_benchmark_on_path()

    from utils.conversation_creator import ConversationCreator

    agent_config = load_yaml_config(agent_config_path)
    dataset_config = load_yaml_config(dataset_config_path)

    old_cwd = os.getcwd()
    try:
        os.chdir(bench_root)
        creator = ConversationCreator(agent_config, dataset_config)
        contexts = creator.get_chunks()
        query_sets = creator.get_query_and_answers()
    finally:
        os.chdir(old_cwd)
    examples: List[AblationExample] = []
    for context_index, (chunks, queries) in enumerate(zip(contexts, query_sets)):
        if max_contexts is not None and context_index >= max_contexts:
            break
        ingest = [normalize_ingest_item(chunk, i) for i, chunk in enumerate(chunks)]
        normalized_queries: List[Dict[str, Any]] = []
        for query_index, query_data in enumerate(queries):
            if max_queries_per_context is not None and query_index >= max_queries_per_context:
                break
            if len(query_data) == 4:
                query, answer, qa_pair_id, eval_metadata = query_data
            elif len(query_data) == 3:
                query, answer, qa_pair_id = query_data
                eval_metadata = None
            else:
                raise ValueError(f"Unexpected query tuple length: {len(query_data)}")
            normalized = normalize_query_item(
                {
                    "query_id": qa_pair_id or f"q_{query_index}",
                    "query": query,
                    "answer": answer,
                    "eval_metadata": eval_metadata or {},
                },
                query_index,
            )
            normalized_queries.append(normalized)
        examples.append(
            AblationExample(
                context_id=f"context_{context_index}",
                user_id=f"context_{context_index}_{dataset_config.get('sub_dataset', 'benchmark')}",
                ingest=ingest,
                queries=normalized_queries,
                eval_metadata={
                    "dataset": dataset_config.get("dataset"),
                    "sub_dataset": dataset_config.get("sub_dataset"),
                },
            )
        )
    return examples


def workload_group_for_query(query: Dict[str, Any]) -> str:
    explicit = query.get("workload_group")
    if explicit:
        return str(explicit)

    eval_metadata = query.get("eval_metadata") or {}
    category = str(eval_metadata.get("category") or "").strip()
    locomo_map = {
        "1": "W5_multi_hop",
        "2": "W4_temporal_context",
        "3": "W1_semantic_general",
        "4": "W1_semantic_general",
        "5": "W2_keyword_exact",
    }
    if category in locomo_map:
        return locomo_map[category]

    for key in ("question_type", "slice", "branch"):
        value = str(eval_metadata.get(key) or "").lower()
        if "multi" in value or "hop" in value:
            return "W5_multi_hop"
        if "time" in value or "temporal" in value or "date" in value:
            return "W4_temporal_context"
        if "entity" in value:
            return "W3_entity_reference"

    text = str(query.get("query") or "").lower()
    if re.search(r"\b(before|after|when|yesterday|today|tomorrow|last|next|date|time|earlier)\b", text):
        return "W4_temporal_context"
    if re.search(r"[\"'`]|[A-Z]{2,}|[_/.-]", str(query.get("query") or "")):
        return "W2_keyword_exact"
    if re.search(r"\b(he|she|they|his|her|their|it|that|this|小红|小明|他|她|它|他们|她们)\b", text):
        return "W3_entity_reference"
    if re.search(r"\b(and|both|relationship|compare|connect|关联|关系|同时|分别)\b", text):
        return "W5_multi_hop"
    return "W1_semantic_general"


def expected_source_ids(query: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ("gold_source_ids", "target_source_ids", "source_ids", "evidence"):
        ids.extend(_flatten_strings(query.get(key)))
    eval_metadata = query.get("eval_metadata") or {}
    for key in ("target_source_ids", "evidence", "source_ids"):
        ids.extend(_flatten_strings(eval_metadata.get(key)))
    return _dedupe(ids)


def retrieved_source_groups(results: List[Dict[str, Any]]) -> List[List[str]]:
    groups: List[List[str]] = []
    for item in results:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        ids = _source_ids_from_metadata(metadata)
        if not ids:
            ids = _source_ids_from_metadata(item)
        if ids:
            groups.append(ids)
    return groups


def _memory_text(item: Dict[str, Any]) -> str:
    return str(
        item.get("memory")
        or item.get("data")
        or item.get("text")
        or item.get("content")
        or ""
    )


def _memory_id(item: Dict[str, Any]) -> str:
    return str(item.get("id") or item.get("memory_id") or "")


def _memory_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return dict(metadata)


def _memory_source_ids(item: Dict[str, Any]) -> List[str]:
    metadata = _memory_metadata(item)
    ids = _source_ids_from_metadata(metadata)
    if not ids:
        ids = _source_ids_from_metadata(item)
    return ids


def _memory_chunk_id(item: Dict[str, Any]) -> str:
    metadata = _memory_metadata(item)
    return str(item.get("chunk_id") or metadata.get("chunk_id") or "")


def _memory_results(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def make_written_memory_rows(
    *,
    config_id: str,
    changed_technique: str,
    context_id: str,
    user_id: str,
    memories_payload: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(_memory_results(memories_payload)):
        metadata = _memory_metadata(item)
        rows.append(
            {
                "config": config_id,
                "changed_technique": changed_technique,
                "context_id": context_id,
                "user_id": user_id,
                "memory_index": index,
                "memory_id": _memory_id(item),
                "memory": _memory_text(item),
                "source_ids": _memory_source_ids(item),
                "chunk_id": _memory_chunk_id(item),
                "score": item.get("score"),
                "hash": item.get("hash"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "metadata": metadata,
            }
        )
    return rows


def make_retrieved_context_rows(
    *,
    config_id: str,
    changed_technique: str,
    context_id: str,
    query: Dict[str, Any],
    query_row: Dict[str, Any],
    workload_group: str,
    search_result: Dict[str, Any],
    final_answer: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    query_id = str(query.get("query_id") or query_row.get("query_id") or "")
    gold_answer = query.get("answer") or query.get("gold_answer")
    results = _memory_results(search_result)
    for rank, item in enumerate(results, start=1):
        metadata = _memory_metadata(item)
        rows.append(
            {
                "config": config_id,
                "changed_technique": changed_technique,
                "context_id": context_id,
                "query_id": query_id,
                "workload_group": workload_group,
                "rank": rank,
                "query": query.get("query", ""),
                "gold_answer": gold_answer,
                "final_answer": final_answer,
                "accuracy": query_row.get("accuracy"),
                "accuracy_metric": query_row.get("accuracy_metric"),
                "retrieved_memory_count": query_row.get("retrieved_memory_count"),
                "retrieved_context_tokens": query_row.get("retrieved_context_tokens"),
                "memory_id": _memory_id(item),
                "memory": _memory_text(item),
                "score": item.get("score"),
                "source_ids": _memory_source_ids(item),
                "chunk_id": _memory_chunk_id(item),
                "metadata": metadata,
            }
        )
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def source_recall_at_k(groups: List[List[str]], targets: List[str], k: int) -> float:
    target_set = set(targets)
    if not target_set:
        return 0.0
    covered = set()
    for group in groups[:k]:
        covered.update(group)
    return len(covered & target_set) / len(target_set)


def normalize_answer(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text or "").lower())).strip()


def token_f1(prediction: Any, answer: Any) -> float:
    predicted = normalize_answer(prediction).split()
    answers = _flatten_strings(answer)
    if not answers:
        return 0.0
    best = 0.0
    for gold in answers:
        gold_tokens = normalize_answer(gold).split()
        if not predicted or not gold_tokens:
            score = 1.0 if predicted == gold_tokens else 0.0
        else:
            common = set(predicted) & set(gold_tokens)
            if not common:
                score = 0.0
            else:
                precision = len(common) / len(predicted)
                recall = len(common) / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


class OptionalAnswerer:
    def __init__(
        self,
        *,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        usage_callback,
        system_prompt: Optional[str] = None,
        answer_template: Optional[str] = None,
        include_current_time: bool = False,
        timeout: Optional[float] = None,
        continue_on_error: bool = False,
    ) -> None:
        self.model = model
        self.client = None
        self.system_prompt = system_prompt or "Answer the question using only the retrieved memories."
        self.answer_template = answer_template
        self.include_current_time = include_current_time
        self.usage_callback = usage_callback
        self.continue_on_error = continue_on_error
        if model:
            self.client = OpenAILLM(
                LLMConfig(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.0,
                    timeout=float(timeout) if timeout is not None else 60.0,
                )
            )
            self.client.usage_callback = usage_callback

    def answer(self, *, query: str, memories: List[str]) -> str:
        if self.client is None:
            return "\n".join(memories)
        memories_text = "\n".join(memories) or "(No retrieved memories found.)"
        if self.answer_template:
            user_content = self.answer_template.format(memories=memories_text, question=query)
            template_name = "benchmark"
        else:
            user_content = f"Retrieved memories:\n{memories_text}\n\nQuestion:\n{query}"
            template_name = "simple"
        if self.include_current_time:
            user_content = f"{user_content}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        try:
            return self.client.generate_response(
                messages,
                usage_stage="answer.generation",
                usage_extra={
                    "retrieved_memory_count": len(memories),
                    "answer_template": template_name,
                },
            )
        except Exception as e:
            if not self.continue_on_error:
                raise
            if self.usage_callback is not None:
                self.usage_callback(
                    {
                        "stage": "answer.error",
                        "kind": "chat",
                        "chat_input_tokens": 0,
                        "chat_output_tokens": 0,
                        "cached_tokens": 0,
                        "embedding_tokens": 0,
                        "total_tokens": 0,
                        "latency_s": 0,
                        "model": self.model,
                        "usage_missing": True,
                        "extra": {"error": str(e)[:500]},
                    }
                )
            return ""


def clone_config_for_run(
    base_config: LiteMemConfig,
    *,
    output_dir: str,
    config_id: str,
    context_id: str,
    flags: TechniqueFlags,
    usage_callback,
) -> LiteMemConfig:
    cfg = copy.deepcopy(base_config)
    safe_context = re.sub(r"[^A-Za-z0-9_.-]+", "_", context_id)
    runtime_dir = os.path.join(output_dir, "runtime", config_id, safe_context)
    if os.path.exists(runtime_dir):
        shutil.rmtree(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    cfg.vector_store.db_path = os.path.join(runtime_dir, "vectors.db")
    cfg.vector_store.collection_name = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", f"litemem_{config_id}_{safe_context}"
    )
    cfg.history_db_path = os.path.join(runtime_dir, "history.db")
    cfg.technique_flags = copy.deepcopy(flags)
    cfg.usage_callback = usage_callback
    return cfg


def aggregate_events(events: List[Dict[str, Any]]) -> Dict[str, float]:
    totals = defaultdict(float)
    for event in events:
        totals["chat_input_tokens"] += int(event.get("chat_input_tokens") or 0)
        totals["chat_output_tokens"] += int(event.get("chat_output_tokens") or 0)
        totals["cached_tokens"] += int(event.get("cached_tokens") or 0)
        totals["embedding_tokens"] += int(event.get("embedding_tokens") or 0)
        totals["latency_s"] += float(event.get("latency_s") or 0.0)
    return dict(totals)


def filter_events(
    events: List[Dict[str, Any]],
    *,
    config: str,
    context_id: str,
    query_id: Optional[str] = None,
    operation: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        if event.get("config") != config or event.get("context_id") != context_id:
            continue
        if query_id is not None and event.get("query_id") != query_id:
            continue
        if query_id is None and event.get("query_id") is not None:
            continue
        if operation is not None and event.get("operation") != operation:
            continue
        out.append(event)
    return out


def make_query_row(
    *,
    config_id: str,
    changed_technique: str,
    context_id: str,
    query: Dict[str, Any],
    query_index: int,
    workload_group: str,
    search_result: Dict[str, Any],
    final_answer: str,
    answer_generated: bool,
    search_latency: float,
    answer_latency: float,
    context_write_latency: float,
    context_query_count: int,
    ledger: UsageLedger,
) -> Dict[str, Any]:
    query_id = str(query.get("query_id") or f"q_{query_index}")
    results = search_result.get("results", [])
    memories = [r.get("memory", "") for r in results if r.get("memory")]
    retrieved_ids = [str(r.get("id")) for r in results if r.get("id")]
    retrieved_groups = retrieved_source_groups(results)
    targets = expected_source_ids(query)
    recall = source_recall_at_k(retrieved_groups, targets, k=max(len(results), 1)) if targets else 0.0
    rank_score = mrr(retrieved_ids, _flatten_strings(query.get("gold_memory_ids")))
    answer = query.get("answer") or query.get("gold_answer")
    qa_score = token_f1(final_answer, answer) if answer else 0.0
    if answer and answer_generated:
        accuracy = qa_score
        accuracy_metric = "qa_f1"
    elif targets:
        accuracy = recall
        accuracy_metric = "source_recall"
    elif query.get("gold_memory_ids"):
        accuracy = rank_score
        accuracy_metric = "mrr"
    else:
        accuracy = qa_score
        accuracy_metric = "extractive_qa_f1"

    write_events = filter_events(
        ledger.events,
        config=config_id,
        context_id=context_id,
        query_id=None,
    )
    query_events = filter_events(
        ledger.events,
        config=config_id,
        context_id=context_id,
        query_id=query_id,
    )
    answer_events = [e for e in query_events if str(e.get("stage", "")).startswith("answer.")]
    litemem_read_events = [e for e in query_events if not str(e.get("stage", "")).startswith("answer.")]
    write_cost = aggregate_events(write_events)
    read_cost = aggregate_events(litemem_read_events)
    answer_cost = aggregate_events(answer_events)

    divisor = max(int(context_query_count), 1)
    amortized_write_input = write_cost.get("chat_input_tokens", 0) / divisor
    amortized_write_output = write_cost.get("chat_output_tokens", 0) / divisor
    amortized_write_cached = write_cost.get("cached_tokens", 0) / divisor
    amortized_write_embedding = write_cost.get("embedding_tokens", 0) / divisor
    amortized_write_latency = context_write_latency / divisor

    litemem_chat_input = amortized_write_input + read_cost.get("chat_input_tokens", 0)
    litemem_chat_output = amortized_write_output + read_cost.get("chat_output_tokens", 0)
    answer_input = answer_cost.get("chat_input_tokens", 0)
    answer_output = answer_cost.get("chat_output_tokens", 0)
    input_tokens = litemem_chat_input + answer_input
    output_tokens = litemem_chat_output + answer_output
    cached_tokens = (
        amortized_write_cached
        + read_cost.get("cached_tokens", 0)
        + answer_cost.get("cached_tokens", 0)
    )
    embedding_tokens = amortized_write_embedding + read_cost.get("embedding_tokens", 0)
    latency = amortized_write_latency + search_latency + answer_latency
    total_tokens = input_tokens + output_tokens + embedding_tokens

    return {
        "query_id": query_id,
        "context_id": context_id,
        "workload_group": workload_group,
        "config": config_id,
        "changed_technique": changed_technique,
        "accuracy": accuracy,
        "accuracy_metric": accuracy_metric,
        "source_recall": recall,
        "qa_f1": qa_score,
        "mrr": rank_score,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "embedding_tokens": embedding_tokens,
        "total_tokens": total_tokens,
        "latency": latency,
        "litemem_chat_input_tokens": litemem_chat_input,
        "litemem_chat_output_tokens": litemem_chat_output,
        "answer_input_tokens": answer_input,
        "answer_output_tokens": answer_output,
        "non_token_latency": amortized_write_latency + search_latency,
        "answer_latency": answer_latency,
        "answer_generated": answer_generated,
        "retrieved_memory_count": len(results),
        "retrieved_context_tokens": token_count("\n".join(memories)),
        "retrieved_memory_ids": json.dumps(retrieved_ids, ensure_ascii=False),
        "retrieved_source_ids": json.dumps(retrieved_groups, ensure_ascii=False),
        "query": query.get("query", ""),
    }


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_rows(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)

    out: List[Dict[str, Any]] = []
    for key, bucket in sorted(groups.items()):
        item = {group_keys[i]: key[i] for i in range(len(group_keys))}
        item["n_queries"] = len(bucket)
        for metric in (
            "accuracy",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "embedding_tokens",
            "total_tokens",
            "latency",
        ):
            item[metric] = mean(row.get(metric, 0) for row in bucket)
        out.append(item)
    return out


def add_deltas(summary: List[Dict[str, Any]], baseline_keys: List[str]) -> None:
    baselines = {
        tuple(row.get(k) for k in baseline_keys): row
        for row in summary
        if row.get("config") == "C_FULL"
    }
    for row in summary:
        base = baselines.get(tuple(row.get(k) for k in baseline_keys))
        if not base:
            row["delta_accuracy"] = 0.0
            row["delta_total_tokens"] = 0.0
            row["delta_latency"] = 0.0
            continue
        row["delta_accuracy"] = row.get("accuracy", 0) - base.get("accuracy", 0)
        row["delta_total_tokens"] = row.get("total_tokens", 0) - base.get("total_tokens", 0)
        row["delta_latency"] = row.get("latency", 0) - base.get("latency", 0)


def attribution_label(row: Dict[str, Any]) -> str:
    acc = float(row.get("delta_accuracy") or 0)
    tokens = float(row.get("delta_total_tokens") or 0)
    latency = float(row.get("delta_latency") or 0)
    cost = tokens + latency
    eps = 1e-9
    if abs(acc) < eps and abs(cost) < eps:
        return "neutral"
    if acc >= -eps and cost < -eps:
        return "accuracy_gain_cost_down"
    if acc > eps and cost > eps:
        return "accuracy_gain_cost_up"
    if acc < -eps and cost < -eps:
        return "accuracy_drop_cost_down"
    if acc <= eps and cost >= eps:
        return "dominated"
    return "neutral"


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_report(path: str, summary_by_workload: List[Dict[str, Any]]) -> None:
    lines = [
        "# LiteMem Token-Efficient Ablation Report",
        "",
        "## Interpretation",
        "",
    ]
    for row in summary_by_workload:
        if row.get("config") == "C_FULL":
            continue
        label = attribution_label(row)
        row["attribution"] = label
        lines.append(
            f"- {row.get('changed_technique')} on {row.get('workload_group')}: "
            f"accuracy delta {row.get('delta_accuracy', 0):.4f}; "
            f"total tokens delta {row.get('delta_total_tokens', 0):.2f}; "
            f"latency delta {row.get('delta_latency', 0):.4f}s; "
            f"attribution `{label}`."
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Deltas are always relative to `C_FULL` within the same workload group.",
            "- `C_RAW_STORE_NO_L1` is diagnostic and should not be ranked as a normal leave-one-out ablation.",
            "- Provider prices are intentionally not hard-coded in this runner.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_ablation(
    *,
    examples: List[AblationExample],
    output_dir: str,
    base_config: Optional[LiteMemConfig] = None,
    retrieve_num: int = 20,
    answer_model: Optional[str] = None,
    answer_api_key: Optional[str] = None,
    answer_base_url: Optional[str] = None,
    answer_system_prompt: Optional[str] = None,
    answer_template: Optional[str] = None,
    answer_include_current_time: bool = False,
    include_raw_diagnostic: bool = True,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    ledger = UsageLedger(run_id=os.path.basename(os.path.abspath(output_dir)) or "litemem_ablation")
    base_config = base_config or LiteMemConfig()
    query_rows: List[Dict[str, Any]] = []
    written_memory_rows: List[Dict[str, Any]] = []
    retrieved_context_rows: List[Dict[str, Any]] = []

    for config_id, changed_technique, flags in iter_ablation_configs(include_raw_diagnostic):
        for example in examples:
            ledger.set_context(
                config=config_id,
                changed_technique=changed_technique,
                context_id=example.context_id,
                query_id=None,
            )
            cfg = clone_config_for_run(
                base_config,
                output_dir=output_dir,
                config_id=config_id,
                context_id=example.context_id,
                flags=flags,
                usage_callback=ledger.callback,
            )
            mem = LiteMem(cfg)
            answerer = OptionalAnswerer(
                model=answer_model,
                api_key=answer_api_key or cfg.llm.api_key,
                base_url=answer_base_url or cfg.llm.base_url,
                usage_callback=ledger.callback,
                system_prompt=answer_system_prompt,
                answer_template=answer_template,
                include_current_time=answer_include_current_time,
            )
            context_write_start = time.perf_counter()
            try:
                for chunk in example.ingest:
                    content = chunk.get("content", "")
                    metadata = dict(chunk.get("metadata") or {})
                    mem.add(content, user_id=example.user_id, metadata=metadata)
                context_write_latency = time.perf_counter() - context_write_start
                written_memory_rows.extend(
                    make_written_memory_rows(
                        config_id=config_id,
                        changed_technique=changed_technique,
                        context_id=example.context_id,
                        user_id=example.user_id,
                        memories_payload=mem.get_all(
                            filters={"user_id": example.user_id},
                            top_k=100000,
                        ),
                    )
                )

                context_query_count = max(len(example.queries), 1)
                for query_index, query in enumerate(example.queries):
                    query_id = str(query.get("query_id") or f"q_{query_index}")
                    workload_group = workload_group_for_query(query)
                    ledger.set_context(
                        config=config_id,
                        changed_technique=changed_technique,
                        context_id=example.context_id,
                        query_id=query_id,
                    )
                    search_start = time.perf_counter()
                    search_result = mem.search(
                        str(query.get("query") or ""),
                        top_k=retrieve_num,
                        filters={"user_id": example.user_id},
                    )
                    search_latency = time.perf_counter() - search_start
                    memories = [
                        item.get("memory", "")
                        for item in search_result.get("results", [])
                        if item.get("memory")
                    ]
                    answer_start = time.perf_counter()
                    final_answer = answerer.answer(
                        query=str(query.get("query") or ""),
                        memories=memories,
                    )
                    answer_latency = time.perf_counter() - answer_start
                    query_row = make_query_row(
                        config_id=config_id,
                        changed_technique=changed_technique,
                        context_id=example.context_id,
                        query=query,
                        query_index=query_index,
                        workload_group=workload_group,
                        search_result=search_result,
                        final_answer=final_answer,
                        answer_generated=answerer.client is not None,
                        search_latency=search_latency,
                        answer_latency=answer_latency,
                        context_write_latency=context_write_latency,
                        context_query_count=context_query_count,
                        ledger=ledger,
                    )
                    query_rows.append(query_row)
                    retrieved_context_rows.extend(
                        make_retrieved_context_rows(
                            config_id=config_id,
                            changed_technique=changed_technique,
                            context_id=example.context_id,
                            query=query,
                            query_row=query_row,
                            workload_group=workload_group,
                            search_result=search_result,
                            final_answer=final_answer,
                        )
                    )
            finally:
                mem.close()

    summary_by_config = summarize_rows(query_rows, ["config", "changed_technique"])
    summary_by_workload = summarize_rows(
        query_rows,
        ["config", "changed_technique", "workload_group"],
    )
    add_deltas(summary_by_config, [])
    add_deltas(summary_by_workload, ["workload_group"])
    for row in summary_by_config + summary_by_workload:
        row["delta_tokens"] = row.get("delta_total_tokens", 0)
        row["attribution"] = attribution_label(row)

    ledger.write_jsonl(os.path.join(output_dir, "operation_events.jsonl"))
    write_jsonl(os.path.join(output_dir, "written_memories_dump.jsonl"), written_memory_rows)
    write_jsonl(os.path.join(output_dir, "retrieved_context_dump.jsonl"), retrieved_context_rows)
    write_csv(
        os.path.join(output_dir, "summary_by_query.csv"),
        query_rows,
        [
            "query_id",
            "context_id",
            "workload_group",
            "config",
            "changed_technique",
            "accuracy",
            "accuracy_metric",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "embedding_tokens",
            "total_tokens",
            "latency",
            "retrieved_memory_count",
            "retrieved_context_tokens",
            "litemem_chat_input_tokens",
            "litemem_chat_output_tokens",
            "answer_input_tokens",
            "answer_output_tokens",
            "non_token_latency",
            "answer_latency",
            "answer_generated",
            "query",
        ],
    )
    write_csv(
        os.path.join(output_dir, "summary_by_config.csv"),
        summary_by_config,
        [
            "config",
            "changed_technique",
            "n_queries",
            "accuracy",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "embedding_tokens",
            "total_tokens",
            "latency",
            "delta_accuracy",
            "delta_total_tokens",
            "delta_tokens",
            "delta_latency",
            "attribution",
        ],
    )
    write_csv(
        os.path.join(output_dir, "summary_by_config_workload.csv"),
        summary_by_workload,
        [
            "config",
            "changed_technique",
            "workload_group",
            "n_queries",
            "accuracy",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "embedding_tokens",
            "total_tokens",
            "latency",
            "delta_accuracy",
            "delta_total_tokens",
            "delta_tokens",
            "delta_latency",
            "attribution",
        ],
    )
    write_json(os.path.join(output_dir, "summary_by_config.json"), summary_by_config)
    write_json(os.path.join(output_dir, "summary_by_config_workload.json"), summary_by_workload)
    write_json(os.path.join(output_dir, "summary_by_query.json"), query_rows)
    write_report(os.path.join(output_dir, "experiment_report.md"), summary_by_workload)
    return {
        "summary_by_config": summary_by_config,
        "summary_by_config_workload": summary_by_workload,
        "summary_by_query": query_rows,
        "operation_events": ledger.events,
        "written_memories_dump": written_memory_rows,
        "retrieved_context_dump": retrieved_context_rows,
    }


def build_base_config_from_args(args: argparse.Namespace) -> LiteMemConfig:
    return LiteMemConfig(
        llm=LLMConfig(
            model=args.llm_model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=0.0,
        ),
        embedder=EmbedderConfig(
            model=args.embedding_model,
            embedding_dims=args.embedding_dims,
            api_key=args.embedding_api_key or args.api_key,
            base_url=args.embedding_base_url or args.base_url,
        ),
        vector_store=VectorStoreConfig(
            embedding_dims=args.embedding_dims,
            distance_metric="cosine",
        ),
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", help="Path to ablation JSONL examples.")
    source.add_argument("--dataset-config", help="MemoryDataBenchmark dataset YAML.")
    parser.add_argument("--agent-config", help="MemoryDataBenchmark agent YAML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--max-queries-per-context", type=int)
    parser.add_argument("--retrieve-num", type=int)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-dims", type=int, default=1536)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-api-key")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-api-key")
    parser.add_argument("--answer-base-url")
    parser.add_argument(
        "--simple-answer-template",
        action="store_true",
        help="Use the runner's minimal answer prompt instead of MemoryDataBenchmark's benchmark prompt.",
    )
    parser.add_argument("--no-raw-diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.dataset_config and not args.agent_config:
        raise SystemExit("--agent-config is required with --dataset-config")
    agent_config = load_yaml_config(args.agent_config) if args.agent_config else {}
    dataset_config = load_yaml_config(args.dataset_config) if args.dataset_config else {}
    if args.input_jsonl:
        examples = load_jsonl_examples(args.input_jsonl)
    else:
        examples = load_memory_data_benchmark_examples(
            agent_config_path=args.agent_config,
            dataset_config_path=args.dataset_config,
            max_contexts=args.max_contexts,
            max_queries_per_context=args.max_queries_per_context,
        )
    retrieve_num = args.retrieve_num
    if retrieve_num is None:
        retrieve_num = int(agent_config.get("retrieve_num", 20))
    answer_template = None
    answer_system_prompt = None
    answer_include_current_time = False
    if args.dataset_config and not args.simple_answer_template:
        answer_template, answer_system_prompt = load_memory_data_benchmark_answer_prompts(
            agent_config=agent_config,
            dataset_config=dataset_config,
        )
        answer_include_current_time = answer_template is not None
    run_ablation(
        examples=examples,
        output_dir=args.output_dir,
        base_config=build_base_config_from_args(args),
        retrieve_num=retrieve_num,
        answer_model=args.answer_model,
        answer_api_key=args.answer_api_key,
        answer_base_url=args.answer_base_url,
        answer_system_prompt=answer_system_prompt,
        answer_template=answer_template,
        answer_include_current_time=answer_include_current_time,
        include_raw_diagnostic=not args.no_raw_diagnostic,
    )


if __name__ == "__main__":
    main()
