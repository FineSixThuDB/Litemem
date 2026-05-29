"""Token-efficient ablation runner for vendored mem0 V3.

This runner mirrors ``litemem.evaluation.ablation_runner`` but executes the
vendored mem0 V3 pipeline from ``examples/mem0``.  The vendored source is kept
read-only; technique ablations are applied as temporary runtime patches around
``Memory.add`` and ``Memory.search``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
import types
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from litemem.evaluation.ablation_runner import (
    AblationExample,
    OptionalAnswerer,
    UsageLedger,
    attribution_label,
    load_jsonl_examples,
    load_memory_data_benchmark_answer_prompts,
    load_memory_data_benchmark_examples,
    load_yaml_config,
    make_retrieved_context_rows,
    make_query_row,
    make_written_memory_rows,
    summarize_rows,
    workload_group_for_query,
    write_csv,
    write_json,
    write_jsonl,
)
from litemem.evaluation.metrics import token_count


MEM0_CONFIG_TECHNIQUES = {
    "M_FULL": "none",
    "M_MINUS_L2_EXISTING_CONTEXT": "L2_existing_memory_context",
    "M_MINUS_L3_RECENT_MESSAGES": "L3_recent_messages_context",
    "M_MINUS_L5_JSON_RESPONSE_FORMAT": "L5_json_response_format",
    "M_MINUS_R2_BM25_RERANK": "R2_bm25_rerank",
    "M_MINUS_R3_ENTITY_BOOST": "R3_entity_linking_boost",
    "M_MINUS_M6_CUSTOM_EXTRACTION_INSTRUCTION": "M6_custom_extraction_instruction",
    "M_RAW_STORE_NO_L1": "L1_additive_extraction_diagnostic",
}


@dataclass
class Mem0TechniqueFlags:
    use_additive_extraction: bool = True
    use_existing_memory_context: bool = True
    use_recent_messages_context: bool = True
    use_json_response_format: bool = True
    use_bm25: bool = True
    use_entity_boost: bool = True
    use_custom_extraction_instruction: bool = True


def technique_flags_for_config(config_id: str) -> Mem0TechniqueFlags:
    flags = Mem0TechniqueFlags()
    if config_id == "M_MINUS_L2_EXISTING_CONTEXT":
        flags.use_existing_memory_context = False
    elif config_id == "M_MINUS_L3_RECENT_MESSAGES":
        flags.use_recent_messages_context = False
    elif config_id == "M_MINUS_L5_JSON_RESPONSE_FORMAT":
        flags.use_json_response_format = False
    elif config_id == "M_MINUS_R2_BM25_RERANK":
        flags.use_bm25 = False
    elif config_id == "M_MINUS_R3_ENTITY_BOOST":
        flags.use_entity_boost = False
    elif config_id == "M_MINUS_M6_CUSTOM_EXTRACTION_INSTRUCTION":
        flags.use_custom_extraction_instruction = False
    elif config_id == "M_RAW_STORE_NO_L1":
        flags.use_additive_extraction = False
    return flags


def iter_ablation_configs(
    include_raw_diagnostic: bool = True,
    include_custom_instruction: bool = False,
) -> List[Tuple[str, str, Mem0TechniqueFlags]]:
    ids = list(MEM0_CONFIG_TECHNIQUES.keys())
    if not include_raw_diagnostic:
        ids.remove("M_RAW_STORE_NO_L1")
    if not include_custom_instruction:
        ids.remove("M_MINUS_M6_CUSTOM_EXTRACTION_INSTRUCTION")
    return [
        (config_id, MEM0_CONFIG_TECHNIQUES[config_id], technique_flags_for_config(config_id))
        for config_id in ids
    ]


def resolve_mem0_custom_instructions(agent_config: Dict[str, Any]) -> Optional[str]:
    """Return explicit mem0 V3 custom instructions, without benchmark prompt fallback."""
    for key in (
        "mem0_custom_instructions",
        "mem0_v3_custom_instructions",
        "custom_instructions",
        "mem0_fact_extraction_prompt",
    ):
        value = agent_config.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def import_vendored_mem0():
    """Import examples/mem0 as an isolated ``mem0`` package for this process."""
    mem1_root = Path(__file__).resolve().parents[2]
    vendored_pkg = mem1_root / "examples" / "mem0" / "mem0"
    if not vendored_pkg.exists():
        raise RuntimeError(f"Vendored mem0 package not found: {vendored_pkg}")

    for name in list(sys.modules):
        if name == "mem0" or name.startswith("mem0."):
            del sys.modules[name]

    package = types.ModuleType("mem0")
    package.__package__ = "mem0"
    package.__path__ = [str(vendored_pkg)]
    package.__file__ = str(vendored_pkg / "__init__.py")
    package.__version__ = "0.0.0-vendored"
    sys.modules["mem0"] = package

    from mem0.memory.main import Memory  # type: ignore
    import mem0.memory.main as mem0_main  # type: ignore

    return Memory, mem0_main


@contextmanager
def patched_attr(obj: Any, name: str, value: Any):
    sentinel = object()
    old = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, old)


def _usage_attr(obj: Any, *names: str) -> int:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _cached_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)
    if details is None:
        return 0
    return _usage_attr(details, "cached_tokens")


def _estimate_embedding_tokens(kwargs: Dict[str, Any]) -> int:
    raw_input = kwargs.get("input", "")
    if isinstance(raw_input, str):
        return token_count(raw_input)
    if isinstance(raw_input, list):
        return sum(token_count(str(item)) for item in raw_input)
    return token_count(str(raw_input))


def _add_qwen3_no_thinking(params: Dict[str, Any], model: str) -> None:
    if "qwen3" not in str(model or "").lower():
        return
    extra_body = dict(params.get("extra_body") or {})
    extra_body.setdefault("enable_thinking", False)
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("enable_thinking", False)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    params["extra_body"] = extra_body


def instrument_mem0_usage(
    memory: Any,
    ledger: UsageLedger,
    model: str,
    request_timeout: Optional[float] = None,
) -> None:
    """Patch mem0's OpenAI clients to emit token/latency events."""
    if getattr(memory, "_ablation_usage_instrumented", False):
        return
    memory._ablation_usage_instrumented = True

    llm_create = memory.llm.client.chat.completions.create

    def chat_create_with_usage(*args, **kwargs):
        _add_qwen3_no_thinking(kwargs, model)
        if request_timeout is not None:
            kwargs.setdefault("timeout", request_timeout)
        start = time.perf_counter()
        response = llm_create(*args, **kwargs)
        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        ledger.callback(
            {
                "kind": "chat",
                "stage": "add.memory_extraction",
                "chat_input_tokens": _usage_attr(usage, "prompt_tokens", "input_tokens"),
                "chat_output_tokens": _usage_attr(usage, "completion_tokens", "output_tokens"),
                "cached_tokens": _cached_tokens(usage),
                "total_tokens": _usage_attr(usage, "total_tokens"),
                "latency_s": latency,
                "model": kwargs.get("model"),
                "usage_missing": usage is None,
            }
        )
        return response

    memory.llm.client.chat.completions.create = chat_create_with_usage

    if not hasattr(memory.embedding_model, "client"):
        return

    stage_stack: List[str] = []
    embeddings_create = memory.embedding_model.client.embeddings.create

    def embedding_create_with_usage(*args, **kwargs):
        if request_timeout is not None:
            kwargs.setdefault("timeout", request_timeout)
        start = time.perf_counter()
        response = embeddings_create(*args, **kwargs)
        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        total_tokens = _usage_attr(usage, "total_tokens", "prompt_tokens")
        if not total_tokens:
            total_tokens = _estimate_embedding_tokens(kwargs)
        ledger.callback(
            {
                "kind": "embedding",
                "stage": stage_stack[-1] if stage_stack else "embedding.unknown",
                "embedding_tokens": total_tokens,
                "total_tokens": total_tokens,
                "latency_s": latency,
                "model": kwargs.get("model"),
                "usage_missing": usage is None,
            }
        )
        return response

    memory.embedding_model.client.embeddings.create = embedding_create_with_usage

    original_embed = memory.embedding_model.embed
    original_embed_batch = memory.embedding_model.embed_batch

    def stage_for_action(memory_action: Optional[str], batch: bool = False) -> str:
        query_id = ledger.context.get("query_id")
        if query_id is not None:
            if memory_action == "search":
                return "search.semantic_embedding"
            if memory_action == "update":
                return "update.memory_embedding"
            return "search.embedding"
        if memory_action == "search":
            return "add.existing_memory_lookup.embedding"
        if memory_action == "update":
            return "update.memory_embedding"
        return "add.embedding_batch" if batch else "add.embedding"

    def embed_with_stage(text, memory_action=None):
        stage_stack.append(stage_for_action(memory_action, batch=False))
        try:
            return original_embed(text, memory_action)
        finally:
            stage_stack.pop()

    def embed_batch_with_stage(texts, memory_action="add"):
        stage_stack.append(stage_for_action(memory_action, batch=True))
        try:
            return original_embed_batch(texts, memory_action)
        finally:
            stage_stack.pop()

    memory.embedding_model.embed = embed_with_stage
    memory.embedding_model.embed_batch = embed_batch_with_stage


def build_mem0_config(
    *,
    args: argparse.Namespace,
    agent_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    output_dir: str,
    config_id: str,
    context_id: str,
    flags: Mem0TechniqueFlags,
) -> Dict[str, Any]:
    safe_context = re.sub(r"[^A-Za-z0-9_.-]+", "_", context_id)
    runtime_dir = os.path.join(output_dir, "runtime", config_id, safe_context)
    if os.path.exists(runtime_dir):
        shutil.rmtree(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)

    embedding_model = args.embedding_model or agent_config.get("mem0_embedder_model") or "text-embedding-3-small"
    embedding_dims = int(args.embedding_dims or (2560 if "Qwen3-Embedding-4B" in embedding_model else 1536))
    base_url = args.base_url or agent_config.get("base_url")
    embedding_base_url = args.embedding_base_url or agent_config.get("embedding_base_url") or base_url
    custom_instructions = resolve_mem0_custom_instructions(agent_config)

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": args.llm_model or agent_config.get("mem0_llm_model") or agent_config.get("model"),
                "temperature": float(agent_config.get("temperature", 0.0)),
                "max_tokens": int(agent_config.get("mem0_llm_max_tokens", 8192)),
                "api_key": args.api_key,
                "openai_base_url": base_url,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
                "embedding_dims": embedding_dims,
                "api_key": args.embedding_api_key or args.api_key,
                "openai_base_url": embedding_base_url,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": os.path.join(runtime_dir, "qdrant"),
                "collection_name": re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    f"mem0_{dataset_config.get('sub_dataset', 'benchmark')}_{config_id}_{safe_context}",
                ),
                "embedding_model_dims": embedding_dims,
                "on_disk": True,
            },
        },
        "history_db_path": os.path.join(runtime_dir, "history.db"),
        "version": "v1.1",
    }
    if flags.use_custom_extraction_instruction and custom_instructions is not None:
        config["custom_instructions"] = custom_instructions
    return config


class Mem0AblationAdapter:
    def __init__(
        self,
        *,
        memory: Any,
        mem0_main: Any,
        flags: Mem0TechniqueFlags,
        add_infer: bool,
    ) -> None:
        self.memory = memory
        self.mem0_main = mem0_main
        self.flags = flags
        self.add_infer = add_infer

    def add_chunk(self, content: str, *, user_id: str, metadata: Optional[Dict[str, Any]] = None):
        with ExitStack() as stack:
            if not self.flags.use_existing_memory_context:
                stack.enter_context(patched_attr(self.memory.vector_store, "search", lambda *a, **k: []))
                original_embed = self.memory.embedding_model.embed

                def embed_without_existing_lookup(text, memory_action=None):
                    if memory_action == "search":
                        return []
                    return original_embed(text, memory_action)

                stack.enter_context(patched_attr(self.memory.embedding_model, "embed", embed_without_existing_lookup))
            if not self.flags.use_recent_messages_context:
                stack.enter_context(patched_attr(self.memory.db, "get_last_messages", lambda *a, **k: []))
            if not self.flags.use_json_response_format:
                original_generate = self.memory.llm.generate_response

                def generate_without_json(*args, **kwargs):
                    kwargs.pop("response_format", None)
                    return original_generate(*args, **kwargs)

                stack.enter_context(patched_attr(self.memory.llm, "generate_response", generate_without_json))
            if not self.flags.use_entity_boost:
                stack.enter_context(
                    patched_attr(
                        self.mem0_main,
                        "extract_entities_batch",
                        lambda texts: [[] for _ in texts],
                    )
                )
            return self.memory.add(
                [{"role": "user", "content": content}],
                user_id=user_id,
                metadata=metadata,
                infer=self.add_infer and self.flags.use_additive_extraction,
            )

    def search(self, query: str, *, user_id: str, top_k: int, threshold: float):
        with ExitStack() as stack:
            if not self.flags.use_bm25 and hasattr(self.memory.vector_store, "keyword_search"):
                stack.enter_context(patched_attr(self.memory.vector_store, "keyword_search", lambda *a, **k: None))
            if not self.flags.use_entity_boost:
                stack.enter_context(patched_attr(self.memory, "_compute_entity_boosts", lambda *a, **k: {}))
                stack.enter_context(patched_attr(self.mem0_main, "extract_entities", lambda *a, **k: []))
            return self.memory.search(
                query=query,
                top_k=top_k,
                filters={"user_id": user_id},
                threshold=threshold,
            )

    def close(self) -> None:
        for store_name in ("vector_store", "_entity_store"):
            store = getattr(self.memory, store_name, None)
            client = getattr(store, "client", None)
            close = getattr(client, "close", None)
            if callable(close):
                close()


def add_mem0_deltas(summary: List[Dict[str, Any]], baseline_keys: List[str]) -> None:
    baselines = {
        tuple(row.get(k) for k in baseline_keys): row
        for row in summary
        if row.get("config") == "M_FULL"
    }
    for row in summary:
        base = baselines.get(tuple(row.get(k) for k in baseline_keys))
        if not base:
            row["delta_accuracy"] = 0.0
            row["delta_total_tokens"] = 0.0
            row["delta_tokens"] = 0.0
            row["delta_latency"] = 0.0
            continue
        row["delta_accuracy"] = row.get("accuracy", 0) - base.get("accuracy", 0)
        row["delta_total_tokens"] = row.get("total_tokens", 0) - base.get("total_tokens", 0)
        row["delta_tokens"] = row["delta_total_tokens"]
        row["delta_latency"] = row.get("latency", 0) - base.get("latency", 0)


def write_mem0_report(path: str, summary_by_workload: List[Dict[str, Any]]) -> None:
    lines = [
        "# mem0 V3 Token-Efficient Ablation Report",
        "",
        "## Interpretation",
        "",
    ]
    for row in summary_by_workload:
        if row.get("config") == "M_FULL":
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
            "- Deltas are always relative to `M_FULL` within the same workload group.",
            "- `M_RAW_STORE_NO_L1` is diagnostic and should not be ranked as a normal leave-one-out ablation.",
            "- `M_MINUS_M6_CUSTOM_EXTRACTION_INSTRUCTION` is emitted only when explicit mem0 custom instructions are configured.",
            "- Provider prices are intentionally not hard-coded in this runner.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_mem0_outputs(
    *,
    output_dir: str,
    ledger: UsageLedger,
    query_rows: List[Dict[str, Any]],
    written_memory_rows: List[Dict[str, Any]],
    retrieved_context_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ledger.write_jsonl(os.path.join(output_dir, "operation_events.jsonl"))
    write_jsonl(os.path.join(output_dir, "written_memories_dump.jsonl"), written_memory_rows)
    write_jsonl(os.path.join(output_dir, "retrieved_context_dump.jsonl"), retrieved_context_rows)
    summary_by_config = summarize_rows(query_rows, ["config", "changed_technique"])
    summary_by_workload = summarize_rows(query_rows, ["config", "changed_technique", "workload_group"])
    add_mem0_deltas(summary_by_config, [])
    add_mem0_deltas(summary_by_workload, ["workload_group"])
    for row in summary_by_config + summary_by_workload:
        row["attribution"] = attribution_label(row)

    query_fields = [
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
    ]
    write_csv(os.path.join(output_dir, "summary_by_query.csv"), query_rows, query_fields)
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
    write_mem0_report(os.path.join(output_dir, "experiment_report.md"), summary_by_workload)
    return {
        "summary_by_config": summary_by_config,
        "summary_by_config_workload": summary_by_workload,
        "summary_by_query": query_rows,
        "operation_events": ledger.events,
        "written_memories_dump": written_memory_rows,
        "retrieved_context_dump": retrieved_context_rows,
    }


def run_mem0_ablation(
    *,
    examples: List[AblationExample],
    output_dir: str,
    agent_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    args: argparse.Namespace,
    answer_template: Optional[str],
    answer_system_prompt: Optional[str],
    answer_include_current_time: bool,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    mem0_home = os.path.join(output_dir, "runtime", "mem0_home")
    os.makedirs(mem0_home, exist_ok=True)
    os.environ["MEM0_DIR"] = mem0_home
    os.environ["MEM0_TELEMETRY"] = "False"
    Memory, mem0_main = import_vendored_mem0()
    ledger = UsageLedger(run_id=os.path.basename(os.path.abspath(output_dir)) or "mem0_ablation")
    query_rows: List[Dict[str, Any]] = []
    written_memory_rows: List[Dict[str, Any]] = []
    retrieved_context_rows: List[Dict[str, Any]] = []
    retrieve_num = int(args.retrieve_num or agent_config.get("retrieve_num", 20))
    threshold = float(args.threshold)
    add_infer = bool(agent_config.get("mem0_add_infer", True))
    has_custom_instructions = resolve_mem0_custom_instructions(agent_config) is not None
    only_configs = {
        item.strip()
        for item in str(args.only_configs or "").split(",")
        if item.strip()
    }
    completed_queries = 0

    configs = iter_ablation_configs(
        include_raw_diagnostic=not args.no_raw_diagnostic,
        include_custom_instruction=has_custom_instructions,
    )
    if only_configs:
        configs = [item for item in configs if item[0] in only_configs]
    for config_id, changed_technique, flags in configs:
        for example in examples:
            ledger.set_context(
                config=config_id,
                changed_technique=changed_technique,
                context_id=example.context_id,
                query_id=None,
            )
            mem_config = build_mem0_config(
                args=args,
                agent_config=agent_config,
                dataset_config=dataset_config,
                output_dir=output_dir,
                config_id=config_id,
                context_id=example.context_id,
                flags=flags,
            )
            memory = Memory.from_config(mem_config)
            instrument_mem0_usage(
                memory,
                ledger,
                mem_config["llm"]["config"]["model"],
                request_timeout=args.api_timeout,
            )
            mem = Mem0AblationAdapter(
                memory=memory,
                mem0_main=mem0_main,
                flags=flags,
                add_infer=add_infer,
            )
            answerer = OptionalAnswerer(
                model=args.answer_model,
                api_key=args.answer_api_key or args.api_key,
                base_url=args.answer_base_url or args.base_url,
                usage_callback=ledger.callback,
                system_prompt=answer_system_prompt,
                answer_template=answer_template,
                include_current_time=answer_include_current_time,
                timeout=args.answer_timeout or args.api_timeout,
                continue_on_error=args.continue_on_answer_error,
            )
            context_write_start = time.perf_counter()
            try:
                for chunk in example.ingest:
                    mem.add_chunk(
                        chunk.get("content", ""),
                        user_id=example.user_id,
                        metadata=dict(chunk.get("metadata") or {}),
                    )
                context_write_latency = time.perf_counter() - context_write_start
                written_memory_rows.extend(
                    make_written_memory_rows(
                        config_id=config_id,
                        changed_technique=changed_technique,
                        context_id=example.context_id,
                        user_id=example.user_id,
                        memories_payload=memory.get_all(
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
                        user_id=example.user_id,
                        top_k=retrieve_num,
                        threshold=threshold,
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
                        answer_generated=bool(args.answer_model),
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
                    completed_queries += 1
                    if args.flush_every_queries and completed_queries % args.flush_every_queries == 0:
                        write_mem0_outputs(
                            output_dir=output_dir,
                            ledger=ledger,
                            query_rows=query_rows,
                            written_memory_rows=written_memory_rows,
                            retrieved_context_rows=retrieved_context_rows,
                        )
            finally:
                mem.close()

    return write_mem0_outputs(
        output_dir=output_dir,
        ledger=ledger,
        query_rows=query_rows,
        written_memory_rows=written_memory_rows,
        retrieved_context_rows=retrieved_context_rows,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl")
    source.add_argument("--dataset-config")
    parser.add_argument("--agent-config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--max-queries-per-context", type=int)
    parser.add_argument("--retrieve-num", type=int)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--llm-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dims", type=int, default=1536)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-api-key")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-api-key")
    parser.add_argument("--answer-base-url")
    parser.add_argument("--api-timeout", type=float, default=45.0)
    parser.add_argument("--answer-timeout", type=float)
    parser.add_argument("--continue-on-answer-error", action="store_true")
    parser.add_argument("--flush-every-queries", type=int, default=10)
    parser.add_argument("--only-configs", help="Comma-separated mem0 config ids to run, e.g. M_FULL,M_MINUS_R2_BM25_RERANK.")
    parser.add_argument("--simple-answer-template", action="store_true")
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

    answer_template = None
    answer_system_prompt = None
    answer_include_current_time = False
    if args.dataset_config and not args.simple_answer_template:
        answer_template, answer_system_prompt = load_memory_data_benchmark_answer_prompts(
            agent_config=agent_config,
            dataset_config=dataset_config,
        )
        answer_include_current_time = answer_template is not None

    run_mem0_ablation(
        examples=examples,
        output_dir=args.output_dir,
        agent_config=agent_config,
        dataset_config=dataset_config,
        args=args,
        answer_template=answer_template,
        answer_system_prompt=answer_system_prompt,
        answer_include_current_time=answer_include_current_time,
    )


if __name__ == "__main__":
    main()
