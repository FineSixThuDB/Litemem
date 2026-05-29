"""Plot enabled-technique ablation deltas by original LoCoMo category.

This consumes one or more ``summary_by_original_category.csv`` files produced
by ``inspect_ablation_cases.py`` and renders dependency-free SVG bar charts.
The deltas are already enabled-technique effects:

    FULL - MINUS_X

So positive accuracy means the technique helps that category, while positive
token/latency means it costs more.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from litemem.evaluation.plot_ablation_impacts import (
    TECHNIQUE_LABELS,
    ordered_unique,
    render_grouped_bar_svg,
    split_csv,
)


CATEGORY_ORDER = ["multi-hop", "temporal", "open-domain", "single-hop"]


METRICS = {
    "accuracy": {
        "field": "enabled_delta_accuracy",
        "scale": 100.0,
        "suffix": " pp",
        "title": "Accuracy Delta By LoCoMo Category",
        "ylabel": "Accuracy Delta",
        "filename": "enabled_delta_accuracy_by_locomo_category.svg",
        "subtitle": "Enabled technique effect: FULL - MINUS_X. + helps accuracy; - hurts accuracy.",
    },
    "tokens": {
        "field": "enabled_delta_total_tokens",
        "scale": 1.0,
        "suffix": "",
        "title": "Token Delta By LoCoMo Category",
        "ylabel": "Total Token Delta / Query",
        "filename": "enabled_delta_tokens_by_locomo_category.svg",
        "subtitle": "Enabled technique effect: FULL - MINUS_X. + uses more tokens; - saves tokens.",
    },
    "latency": {
        "field": "enabled_delta_latency",
        "scale": 1.0,
        "suffix": "s",
        "title": "Latency Delta By LoCoMo Category",
        "ylabel": "Latency Delta / Query",
        "filename": "enabled_delta_latency_by_locomo_category.svg",
        "subtitle": "Enabled technique effect: FULL - MINUS_X. + is slower; - is faster.",
    },
}


def read_rows(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_metric_data(rows: List[Dict[str, str]], metric: str) -> Dict[str, Dict[str, float]]:
    spec = METRICS[metric]
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        technique = row.get("technique", "")
        category = row.get("original_category_label", "") or row.get("original_category", "")
        if not technique or not category:
            continue
        value = float(row.get(spec["field"]) or 0.0) * spec["scale"]
        out.setdefault(technique, {})[category] = value
    return out


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-files",
        required=True,
        help="Comma-separated summary_by_original_category.csv files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--techniques", help="Comma-separated technique names to include.")
    parser.add_argument("--categories", help="Comma-separated category labels to include.")
    parser.add_argument("--label-bars", action="store_true")
    args = parser.parse_args(argv)

    summary_files = [Path(item) for item in split_csv(args.summary_files)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(summary_files)

    selected_techniques = split_csv(args.techniques) or [
        item
        for item in TECHNIQUE_LABELS
        if any(row.get("technique") == item for row in rows)
    ] or ordered_unique(row.get("technique", "") for row in rows if row.get("technique"))
    selected_categories = split_csv(args.categories) or [
        item
        for item in CATEGORY_ORDER
        if any(row.get("original_category_label") == item for row in rows)
    ]
    rows = [
        row
        for row in rows
        if row.get("technique") in selected_techniques
        and (row.get("original_category_label") or row.get("original_category")) in selected_categories
    ]
    if not rows:
        raise SystemExit("No rows left after filtering.")

    write_csv(
        output_dir / "combined_summary_by_original_category.csv",
        rows,
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
    for metric, spec in METRICS.items():
        render_grouped_bar_svg(
            data=build_metric_data(rows, metric),
            techniques=selected_techniques,
            workloads=selected_categories,
            title=spec["title"],
            subtitle=spec["subtitle"],
            ylabel=spec["ylabel"],
            suffix=spec["suffix"],
            output_path=output_dir / spec["filename"],
            label_bars=args.label_bars,
        )
        print(output_dir / spec["filename"])
    print(output_dir / "combined_summary_by_original_category.csv")


if __name__ == "__main__":
    main()
