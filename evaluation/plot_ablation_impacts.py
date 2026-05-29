"""Plot token-efficient ablation impacts as dependency-free SVG bar charts.

The runner summaries store leave-one-out deltas as:

    C_MINUS_X / M_MINUS_X - FULL

This script can visualize either that removal effect directly, or flip the sign
to show the estimated effect of enabling a technique:

    FULL - MINUS_X
"""

from __future__ import annotations

import argparse
import csv
import math
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
]


WORKLOAD_LABELS = {
    "W1_semantic_general": "W1 semantic",
    "W4_temporal_context": "W4 temporal",
    "W5_multi_hop": "W5 multi-hop",
}


TECHNIQUE_LABELS = {
    "L2_existing_memory_context": "L2 existing context",
    "L3_recent_messages_context": "L3 recent messages",
    "L5_json_response_format": "L5 JSON format",
    "R2_bm25_rerank": "R2 BM25 rerank",
    "R3_entity_linking_boost": "R3 entity boost",
    "L1_additive_extraction_diagnostic": "L1 raw-store diagnostic",
}


METRICS = {
    "accuracy": {
        "field": "delta_accuracy",
        "scale": 100.0,
        "suffix": " pp",
        "title": "Accuracy Delta",
        "explain_enabled": "+ means technique improves accuracy; - means technique hurts accuracy",
        "explain_removal": "+ means removal improves accuracy; - means removal hurts accuracy",
        "filename": "ablation_delta_accuracy_by_workload.svg",
    },
    "tokens": {
        "field": "delta_total_tokens",
        "scale": 1.0,
        "suffix": "",
        "title": "Total Token Delta / Query",
        "explain_enabled": "+ means technique uses more tokens; - means technique saves tokens",
        "explain_removal": "+ means removal uses more tokens; - means removal saves tokens",
        "filename": "ablation_delta_total_tokens_by_workload.svg",
    },
    "latency": {
        "field": "delta_latency",
        "scale": 1.0,
        "suffix": "s",
        "title": "Latency Delta / Query",
        "explain_enabled": "+ means technique is slower; - means technique is faster",
        "explain_removal": "+ means removal is slower; - means removal is faster",
        "filename": "ablation_delta_latency_by_workload.svg",
    },
}


def read_rows(input_dir: Path, include_diagnostic: bool) -> List[Dict[str, str]]:
    path = input_dir / "summary_by_config_workload.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        config = row.get("config", "")
        technique = row.get("changed_technique", "")
        if technique == "none":
            continue
        if not include_diagnostic and (
            "RAW_STORE_NO_L1" in config or "L1_additive_extraction_diagnostic" in technique
        ):
            continue
        out.append(row)
    return out


def split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if math.isclose(vmin, vmax):
        pad = abs(vmin) * 0.1 or 1.0
        vmin -= pad
        vmax += pad
    span = vmax - vmin
    raw_step = span / max(count - 1, 1)
    exponent = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
    base = 10**exponent
    fraction = raw_step / base
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    step = nice_fraction * base
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(0.0 if abs(value) < 1e-12 else value)
        value += step
    if 0.0 not in ticks and start < 0 < end:
        ticks.append(0.0)
        ticks.sort()
    return ticks


def fmt_value(value: float, suffix: str) -> str:
    if suffix == " pp":
        return f"{value:+.1f}{suffix}"
    if suffix == "s":
        return f"{value:+.2f}{suffix}"
    return f"{value:+.0f}{suffix}"


def display_workload(value: str) -> str:
    return WORKLOAD_LABELS.get(value, value)


def display_technique(value: str) -> str:
    return TECHNIQUE_LABELS.get(value, value)


def render_grouped_bar_svg(
    *,
    data: Dict[str, Dict[str, float]],
    techniques: Sequence[str],
    workloads: Sequence[str],
    title: str,
    subtitle: str,
    ylabel: str,
    suffix: str,
    output_path: Path,
    label_bars: bool,
) -> None:
    group_width = 220
    bar_width = 26
    bar_gap = 8
    left = 92
    right = 44
    top = 116
    bottom = 86
    plot_h = 360
    width = left + right + max(1, len(techniques)) * group_width
    height = top + plot_h + bottom

    values = [
        data.get(technique, {}).get(workload, 0.0)
        for technique in techniques
        for workload in workloads
    ] or [0.0]
    vmin = min(values + [0.0])
    vmax = max(values + [0.0])
    pad = (vmax - vmin) * 0.12 or 1.0
    ticks = nice_ticks(vmin - pad, vmax + pad)
    y_min = min(ticks)
    y_max = max(ticks)
    if math.isclose(y_min, y_max):
        y_min -= 1
        y_max += 1

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    zero_y = y(0.0)
    colors = {workload: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, workload in enumerate(workloads)}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;fill:#111827}",
        ".sub{fill:#4b5563;font-size:13px}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}",
        ".label{font-size:12px}.small{font-size:11px;fill:#374151}.title{font-size:20px;font-weight:700}",
        ".legend{font-size:12px;fill:#111827}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" class="title">{escape(title)}</text>',
        f'<text x="{left}" y="52" class="sub">{escape(subtitle)}</text>',
    ]

    legend_y = 82
    legend_x = left
    for workload in workloads:
        label = display_workload(workload)
        lines.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="12" height="12" fill="{colors[workload]}" rx="2"/>')
        lines.append(f'<text x="{legend_x+18}" y="{legend_y}" class="legend">{escape(label)}</text>')
        legend_x += 18 + max(92, len(label) * 8)

    for tick in ticks:
        ty = y(tick)
        lines.append(f'<line x1="{left}" y1="{ty:.2f}" x2="{width-right}" y2="{ty:.2f}" class="grid"/>')
        lines.append(
            f'<text x="{left-10}" y="{ty+4:.2f}" text-anchor="end" class="small">{escape(fmt_value(tick, suffix))}</text>'
        )
    lines.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')

    inner_bar_w = len(workloads) * bar_width + max(0, len(workloads) - 1) * bar_gap
    for i, technique in enumerate(techniques):
        group_x = left + i * group_width
        start_x = group_x + (group_width - inner_bar_w) / 2
        for j, workload in enumerate(workloads):
            value = data.get(technique, {}).get(workload, 0.0)
            x = start_x + j * (bar_width + bar_gap)
            y0 = y(0.0)
            yv = y(value)
            rect_y = min(y0, yv)
            rect_h = max(abs(yv - y0), 1.0)
            lines.append(
                f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_width}" height="{rect_h:.2f}" fill="{colors[workload]}" rx="2"/>'
            )
            if label_bars:
                text_y = rect_y - 5 if value >= 0 else rect_y + rect_h + 13
                lines.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{text_y:.2f}" text-anchor="middle" class="small">{escape(fmt_value(value, suffix))}</text>'
                )
        label = display_technique(technique)
        lines.append(
            f'<text x="{group_x + group_width / 2:.2f}" y="{top + plot_h + 34}" text-anchor="middle" class="label">{escape(label)}</text>'
        )

    lines.append(
        f'<text x="24" y="{top + plot_h / 2:.2f}" text-anchor="middle" class="small" transform="rotate(-90 24 {top + plot_h / 2:.2f})">{escape(ylabel)}</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_metric_data(
    rows: List[Dict[str, str]],
    *,
    metric: str,
    direction: str,
) -> Dict[str, Dict[str, float]]:
    spec = METRICS[metric]
    multiplier = -1.0 if direction == "enabled" else 1.0
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        technique = row.get("changed_technique", "")
        workload = row.get("workload_group", "")
        if not technique or not workload:
            continue
        value = float(row.get(spec["field"]) or 0.0) * spec["scale"] * multiplier
        out.setdefault(technique, {})[workload] = value
    return out


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Ablation output directory.")
    parser.add_argument("--output-dir", help="Directory for SVG charts. Defaults to <input-dir>/figures.")
    parser.add_argument(
        "--direction",
        choices=("removal", "enabled", "precomputed"),
        default="removal",
        help="Plot leave-one-out removal deltas, flip signs to estimate enabled effects, or use precomputed deltas as-is.",
    )
    parser.add_argument(
        "--techniques",
        help="Comma-separated changed_technique names to include, preserving this order.",
    )
    parser.add_argument(
        "--workloads",
        help="Comma-separated workload_group names to include, preserving this order.",
    )
    parser.add_argument("--include-diagnostic", action="store_true", help="Include raw-store/no-L1 diagnostic rows.")
    parser.add_argument("--label-bars", action="store_true", help="Draw value labels above bars.")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_dir, include_diagnostic=args.include_diagnostic)
    selected_techniques = split_csv(args.techniques) or ordered_unique(row["changed_technique"] for row in rows)
    selected_workloads = split_csv(args.workloads) or ordered_unique(row["workload_group"] for row in rows)
    rows = [
        row
        for row in rows
        if row.get("changed_technique") in selected_techniques
        and row.get("workload_group") in selected_workloads
    ]
    techniques = [t for t in selected_techniques if any(row.get("changed_technique") == t for row in rows)]
    workloads = [w for w in selected_workloads if any(row.get("workload_group") == w for row in rows)]
    if not rows or not techniques or not workloads:
        raise SystemExit("No rows left after filtering.")

    direction_text = (
        "Enabled technique effect: FULL - MINUS_X"
        if args.direction == "enabled"
        else "Enabled technique effect: precomputed"
        if args.direction == "precomputed"
        else "Removal effect: MINUS_X - FULL"
    )
    for metric, spec in METRICS.items():
        data = build_metric_data(rows, metric=metric, direction=args.direction)
        path = output_dir / spec["filename"]
        explain_key = "explain_enabled" if args.direction in {"enabled", "precomputed"} else "explain_removal"
        render_grouped_bar_svg(
            data=data,
            techniques=techniques,
            workloads=workloads,
            title=spec["title"],
            subtitle=f"{direction_text}. {spec[explain_key]}",
            ylabel=spec["title"],
            suffix=spec["suffix"],
            output_path=path,
            label_bars=args.label_bars,
        )
        print(path)


if __name__ == "__main__":
    main()
