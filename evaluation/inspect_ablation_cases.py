"""Inspect query-level ablation deltas and dump concrete case studies.

This script is intentionally dependency-free. It reads an ablation output
directory and produces:

- query_level_enabled_deltas.csv
- summary_by_original_category.csv
- ablation_case_report.md

The runner stores leave-one-out rows as MINUS_X. This inspector flips them into
"enabled technique effect" so positive accuracy means FULL is better than
MINUS_X.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from litemem.evaluation.ablation_runner import (
    load_memory_data_benchmark_examples,
    workload_group_for_query,
)


FIELDNAMES = [
    "query_id",
    "context_id",
    "original_category",
    "original_category_label",
    "workload_group",
    "technique",
    "minus_config",
    "enabled_delta_accuracy",
    "enabled_delta_total_tokens",
    "enabled_delta_latency",
    "full_accuracy",
    "minus_accuracy",
    "full_total_tokens",
    "minus_total_tokens",
    "full_latency",
    "minus_latency",
    "query",
    "gold_answer",
]


LOCOMO_CATEGORY_LABELS = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def clean_query(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    marker = "possible."
    idx = text.lower().find(marker)
    if idx >= 0:
        return text[idx + len(marker) :].strip()
    return text


def query_metadata(agent_config: str, dataset_config: str, max_contexts: Optional[int]) -> Dict[str, Dict[str, Any]]:
    examples = load_memory_data_benchmark_examples(
        agent_config_path=agent_config,
        dataset_config_path=dataset_config,
        max_contexts=max_contexts,
        max_queries_per_context=None,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for example in examples:
        for query in example.queries:
            query_id = str(query.get("query_id") or "")
            eval_metadata = query.get("eval_metadata") or {}
            out[query_id] = {
                "query": clean_query(str(query.get("query") or "")),
                "gold_answer": query.get("answer") or query.get("gold_answer") or "",
                "original_category": str(eval_metadata.get("category") or ""),
                "original_category_label": LOCOMO_CATEGORY_LABELS.get(str(eval_metadata.get("category") or ""), ""),
                "workload_group": workload_group_for_query(query),
            }
    return out


def build_context_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("config") or ""), str(row.get("query_id") or ""))
        index[key].append(row)
    for key in index:
        index[key].sort(key=lambda item: as_float(item.get("rank")))
    return index


def top_memories(
    context_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    *,
    config: str,
    query_id: str,
    limit: int,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    rows = context_index.get((config, query_id), [])
    if not rows:
        return "", "", []
    final_answer = str(rows[0].get("final_answer") or "")
    gold_answer = rows[0].get("gold_answer") or ""
    return final_answer, gold_answer, rows[:limit]


def summarize(records: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[tuple(record.get(key) for key in group_keys)].append(record)
    rows = []
    for key, bucket in sorted(buckets.items()):
        row = {group_keys[i]: key[i] for i in range(len(group_keys))}
        row["n_queries"] = len(bucket)
        row["enabled_delta_accuracy"] = mean(as_float(r["enabled_delta_accuracy"]) for r in bucket)
        row["enabled_delta_total_tokens"] = mean(as_float(r["enabled_delta_total_tokens"]) for r in bucket)
        row["enabled_delta_latency"] = mean(as_float(r["enabled_delta_latency"]) for r in bucket)
        rows.append(row)
    return rows


def format_memory_rows(rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for row in rows:
        rank = row.get("rank")
        score = row.get("score")
        memory = re.sub(r"\s+", " ", str(row.get("memory") or "")).strip()
        if len(memory) > 260:
            memory = memory[:257] + "..."
        lines.append(f"  {rank}. score={score} | {memory}")
    return lines


def write_case_report(
    path: Path,
    *,
    records: List[Dict[str, Any]],
    context_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    top_n: int,
    top_k_memories: int,
) -> None:
    lines = [
        "# Ablation Case Report",
        "",
        "Deltas are enabled-technique effects: `FULL - MINUS_X`.",
        "Positive accuracy means the technique helps; negative accuracy means it hurts.",
        "Positive tokens/latency means the technique costs more.",
        "",
    ]
    techniques = sorted({str(r["technique"]) for r in records})
    for technique in techniques:
        bucket = [r for r in records if r["technique"] == technique]
        helpful = sorted(bucket, key=lambda r: as_float(r["enabled_delta_accuracy"]), reverse=True)[:top_n]
        harmful = sorted(bucket, key=lambda r: as_float(r["enabled_delta_accuracy"]))[:top_n]
        for title, cases in (("Most Helpful Cases", helpful), ("Most Harmful Cases", harmful)):
            lines.extend([f"## {technique} - {title}", ""])
            for case in cases:
                delta_acc = as_float(case["enabled_delta_accuracy"]) * 100
                delta_tokens = as_float(case["enabled_delta_total_tokens"])
                delta_latency = as_float(case["enabled_delta_latency"])
                minus_config = str(case["minus_config"])
                query_id = str(case["query_id"])
                full_answer, gold_from_dump, full_context = top_memories(
                    context_index,
                    config="M_FULL",
                    query_id=query_id,
                    limit=top_k_memories,
                )
                minus_answer, _, minus_context = top_memories(
                    context_index,
                    config=minus_config,
                    query_id=query_id,
                    limit=top_k_memories,
                )
                gold = case.get("gold_answer") or gold_from_dump
                lines.extend(
                    [
                        f"### {query_id}",
                        "",
                        f"- category/workload: `{case.get('original_category')}` / `{case.get('workload_group')}`",
                        f"- category label: {case.get('original_category_label')}",
                        f"- query: {case.get('query')}",
                        f"- gold: {gold}",
                        f"- enabled deltas: accuracy {delta_acc:+.1f} pp; tokens {delta_tokens:+.1f}; latency {delta_latency:+.2f}s",
                        f"- FULL accuracy/answer: {as_float(case['full_accuracy']):.4f} / {full_answer}",
                        f"- {minus_config} accuracy/answer: {as_float(case['minus_accuracy']):.4f} / {minus_answer}",
                        "",
                        "FULL top retrieved memories:",
                    ]
                )
                lines.extend(format_memory_rows(full_context))
                lines.extend(["", f"{minus_config} top retrieved memories:"])
                lines.extend(format_memory_rows(minus_context))
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--baseline-dir", help="Optional directory containing M_FULL query/context dumps.")
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--max-contexts", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-k-memories", type=int, default=8)
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    query_rows = read_csv(input_dir / "summary_by_query.csv")
    context_rows = read_jsonl(input_dir / "retrieved_context_dump.jsonl")
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else input_dir
    baseline_query_rows = read_csv(baseline_dir / "summary_by_query.csv")
    if baseline_dir != input_dir:
        context_rows = read_jsonl(baseline_dir / "retrieved_context_dump.jsonl") + context_rows
    metadata = query_metadata(args.agent_config, args.dataset_config, args.max_contexts)
    context_index = build_context_index(context_rows)

    full_by_query = {
        row.get("query_id", ""): row
        for row in baseline_query_rows
        if row.get("config") == "M_FULL"
    }
    records: List[Dict[str, Any]] = []
    for row in query_rows:
        config = row.get("config", "")
        if config == "M_FULL":
            continue
        technique = row.get("changed_technique", "")
        if not technique or technique == "none":
            continue
        query_id = row.get("query_id", "")
        full = full_by_query.get(query_id)
        if not full:
            continue
        meta = metadata.get(query_id, {})
        record = {
            "query_id": query_id,
            "context_id": row.get("context_id", ""),
            "original_category": meta.get("original_category", ""),
            "original_category_label": meta.get("original_category_label", ""),
            "workload_group": meta.get("workload_group") or row.get("workload_group", ""),
            "technique": technique,
            "minus_config": config,
            "enabled_delta_accuracy": as_float(full.get("accuracy")) - as_float(row.get("accuracy")),
            "enabled_delta_total_tokens": as_float(full.get("total_tokens")) - as_float(row.get("total_tokens")),
            "enabled_delta_latency": as_float(full.get("latency")) - as_float(row.get("latency")),
            "full_accuracy": as_float(full.get("accuracy")),
            "minus_accuracy": as_float(row.get("accuracy")),
            "full_total_tokens": as_float(full.get("total_tokens")),
            "minus_total_tokens": as_float(row.get("total_tokens")),
            "full_latency": as_float(full.get("latency")),
            "minus_latency": as_float(row.get("latency")),
            "query": meta.get("query") or clean_query(row.get("query", "")),
            "gold_answer": meta.get("gold_answer", ""),
        }
        records.append(record)

    write_csv(input_dir / "query_level_enabled_deltas.csv", records, FIELDNAMES)
    category_summary = summarize(
        records,
        ["technique", "original_category", "original_category_label", "workload_group"],
    )
    write_csv(
        input_dir / "summary_by_original_category.csv",
        category_summary,
        [
            "technique",
            "original_category",
            "original_category_label",
            "workload_group",
            "n_queries",
            "enabled_delta_accuracy",
            "enabled_delta_total_tokens",
            "enabled_delta_latency",
        ],
    )
    write_case_report(
        input_dir / "ablation_case_report.md",
        records=records,
        context_index=context_index,
        top_n=args.top_n,
        top_k_memories=args.top_k_memories,
    )
    print(input_dir / "query_level_enabled_deltas.csv")
    print(input_dir / "summary_by_original_category.csv")
    print(input_dir / "ablation_case_report.md")


if __name__ == "__main__":
    main()
