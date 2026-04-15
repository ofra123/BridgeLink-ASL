"""Dataset summaries, experiment scaffolds, and report assets."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .clip_dataset import ClipDatasetRecord, load_clip_dataset
from .metrics import ClassificationMetrics, compute_classification_metrics


def build_dataset_summary(records: Iterable[ClipDatasetRecord]) -> dict[str, Any]:
    """Summarize a How2Sign subset manifest for the Space and report."""

    materialized = list(records)
    split_counts = Counter(record.split for record in materialized)
    class_counts = Counter(record.label for record in materialized)
    source_counts = Counter(record.source for record in materialized)
    missing_frames = [record.clip_id for record in materialized if not record.sampled_frames]
    return {
        "dataset": "How2Sign subset",
        "total_clips": len(materialized),
        "num_classes": len(class_counts),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "missing_sampled_frames": missing_frames,
        "example_records": [
            {
                "clip_id": record.clip_id,
                "split": record.split,
                "label": record.label,
                "gloss": list(record.gloss),
                "english": record.english,
                "sampled_frame_count": len(record.sampled_frames),
            }
            for record in materialized[:8]
        ],
    }


def build_comparison_results(records: Iterable[ClipDatasetRecord]) -> dict[str, Any]:
    """Build deterministic comparison results until real models are attached."""

    materialized = list(records)
    labels = tuple(sorted({record.label for record in materialized}))
    rows: list[dict[str, Any]] = []
    cnn_predictions: list[str] = []
    vlm_predictions: list[str] = []
    expected: list[str] = []

    for index, record in enumerate(materialized):
        expected.append(record.label)
        cnn_prediction = record.label if index % 4 != 3 else _next_label(record.label, labels)
        vlm_prediction = record.label if index % 5 != 4 else _next_label(record.label, labels)
        cnn_predictions.append(cnn_prediction)
        vlm_predictions.append(vlm_prediction)
        rows.append(
            {
                "clip_id": record.clip_id,
                "split": record.split,
                "expected_label": record.label,
                "expected_english": record.english,
                "cnn_prediction": cnn_prediction,
                "vlm_prediction": vlm_prediction,
                "cnn_correct": cnn_prediction == record.label,
                "vlm_correct": vlm_prediction == record.label,
                "cnn_latency_seconds": round(0.12 + index * 0.01, 3),
                "vlm_latency_seconds": round(2.8 + index * 0.18, 3),
                "notes": "Scaffolded result. Replace with real CNN/Qwen outputs after running experiments.",
            }
        )

    cnn_metrics = compute_classification_metrics(expected, cnn_predictions, labels=labels)
    vlm_metrics = compute_classification_metrics(expected, vlm_predictions, labels=labels)
    return {
        "status": "scaffolded_until_real_experiments_run",
        "cnn_metrics": cnn_metrics.as_dict(),
        "vlm_metrics": vlm_metrics.as_dict(),
        "rows": rows,
        "comparison_chart_svg": make_model_comparison_svg(cnn_metrics, vlm_metrics),
        "class_distribution_svg": make_bar_chart_svg(
            "How2Sign subset class distribution",
            Counter(record.label for record in materialized),
        ),
        "split_distribution_svg": make_bar_chart_svg(
            "Train/validation/test split",
            Counter(record.split for record in materialized),
        ),
        "confusion_matrix_svg": make_confusion_matrix_svg(cnn_metrics),
    }


def make_bar_chart_svg(title: str, values: Counter[str] | dict[str, int]) -> str:
    """Create a simple inline SVG bar chart without extra dependencies."""

    items = list(values.items())
    width = 860
    row_height = 34
    height = max(130, 70 + row_height * max(len(items), 1))
    max_value = max((count for _, count in items), default=1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f'<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="24" y="34" font-size="22" font-family="Georgia, serif" fill="#1d2b2a">{html.escape(title)}</text>',
    ]
    for index, (label, count) in enumerate(items):
        y = 64 + index * row_height
        bar_width = int((width - 280) * (count / max_value))
        lines.extend(
            [
                f'<text x="24" y="{y + 20}" font-size="13" font-family="Arial" fill="#243331">{html.escape(label)}</text>',
                f'<rect x="250" y="{y + 5}" width="{bar_width}" height="20" rx="5" fill="#2f6f73"/>',
                f'<text x="{260 + bar_width}" y="{y + 20}" font-size="13" font-family="Arial" fill="#243331">{count}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def make_model_comparison_svg(cnn: ClassificationMetrics, vlm: ClassificationMetrics) -> str:
    """Create a compact CNN vs VLM metric chart."""

    metrics = [
        ("Accuracy", cnn.accuracy, vlm.accuracy),
        ("Precision", cnn.macro_precision, vlm.macro_precision),
        ("Recall", cnn.macro_recall, vlm.macro_recall),
        ("F1", cnn.macro_f1, vlm.macro_f1),
    ]
    width = 860
    height = 270
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="24" y="34" font-size="22" font-family="Georgia, serif" fill="#1d2b2a">CNN vs VLM metrics</text>',
        '<text x="610" y="34" font-size="13" font-family="Arial" fill="#2f6f73">CNN</text>',
        '<text x="710" y="34" font-size="13" font-family="Arial" fill="#9b5d2e">VLM</text>',
    ]
    for index, (name, cnn_value, vlm_value) in enumerate(metrics):
        y = 68 + index * 46
        lines.append(f'<text x="24" y="{y + 18}" font-size="15" font-family="Arial" fill="#243331">{name}</text>')
        lines.append(f'<rect x="170" y="{y}" width="{int(cnn_value * 390)}" height="16" rx="4" fill="#2f6f73"/>')
        lines.append(f'<rect x="170" y="{y + 21}" width="{int(vlm_value * 390)}" height="16" rx="4" fill="#9b5d2e"/>')
        lines.append(f'<text x="580" y="{y + 13}" font-size="12" font-family="Arial">{cnn_value:.2f}</text>')
        lines.append(f'<text x="580" y="{y + 34}" font-size="12" font-family="Arial">{vlm_value:.2f}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def make_confusion_matrix_svg(metrics: ClassificationMetrics) -> str:
    """Render a tiny confusion matrix SVG for report assets."""

    labels = metrics.labels[:8]
    cell = 42
    left = 160
    top = 74
    width = left + cell * len(labels) + 40
    height = top + cell * len(labels) + 60
    max_count = max(
        (count for row in metrics.confusion_matrix.values() for count in row.values()),
        default=1,
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="24" y="34" font-size="22" font-family="Georgia, serif" fill="#1d2b2a">CNN confusion matrix</text>',
    ]
    for col, label in enumerate(labels):
        x = left + col * cell
        lines.append(f'<text x="{x}" y="62" font-size="10" font-family="Arial" transform="rotate(-35 {x} 62)">{html.escape(label[:18])}</text>')
    for row, actual in enumerate(labels):
        y = top + row * cell
        lines.append(f'<text x="20" y="{y + 25}" font-size="10" font-family="Arial">{html.escape(actual[:22])}</text>')
        for col, predicted in enumerate(labels):
            x = left + col * cell
            count = metrics.confusion_matrix.get(actual, {}).get(predicted, 0)
            shade = 245 - int(150 * (count / max_count))
            fill = f"rgb({shade},{max(shade - 15, 80)},{max(shade - 20, 70)})"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" fill="{fill}"/>')
            lines.append(f'<text x="{x + 15}" y="{y + 24}" font-size="12" font-family="Arial">{count}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def write_comparison_artifacts(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Write JSON/JSONL/SVG artifacts used by the report and Space."""

    records = load_clip_dataset(manifest_path)
    summary = build_dataset_summary(records)
    comparison = build_comparison_results(records)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "dataset_summary": out / "dataset-summary.json",
        "cnn_metrics": out / "cnn-metrics.json",
        "vlm_metrics": out / "vlm-metrics.json",
        "comparison_results": out / "comparison-results.jsonl",
        "class_distribution": out / "class-distribution.svg",
        "split_distribution": out / "split-distribution.svg",
        "model_comparison": out / "model-comparison.svg",
        "confusion_matrix": out / "cnn-confusion-matrix.svg",
    }
    paths["dataset_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["cnn_metrics"].write_text(json.dumps(comparison["cnn_metrics"], indent=2), encoding="utf-8")
    paths["vlm_metrics"].write_text(json.dumps(comparison["vlm_metrics"], indent=2), encoding="utf-8")
    paths["comparison_results"].write_text(
        "\n".join(json.dumps(row) for row in comparison["rows"]) + "\n",
        encoding="utf-8",
    )
    paths["class_distribution"].write_text(comparison["class_distribution_svg"], encoding="utf-8")
    paths["split_distribution"].write_text(comparison["split_distribution_svg"], encoding="utf-8")
    paths["model_comparison"].write_text(comparison["comparison_chart_svg"], encoding="utf-8")
    paths["confusion_matrix"].write_text(comparison["confusion_matrix_svg"], encoding="utf-8")
    return paths


def report_summary_markdown() -> str:
    """Return a concise report outline aligned to the course rubric."""

    return """## CVPR-Style Report Checklist

1. Abstract: ASL accessibility problem, CNN baseline, Qwen2.5-VL comparison, Space demo.
2. Introduction: communication gap, video understanding challenge, project contributions.
3. Related Work: How2Sign, ASL recognition, CNN action recognition, VLM/video understanding, hand/pose methods.
4. Dataset: How2Sign subset selection, labels, samples, splits, visual examples, limitations.
5. Method: sampled-frame CNN baseline and Qwen2.5-VL prompting/inference.
6. Experiments: train/val/test setup, hyperparameters, metrics, hardware, libraries.
7. Results: accuracy, precision, recall, F1, confusion matrix, latency, qualitative successes/failures.
8. Conclusion: what worked, limitations, and future work.
"""


def _next_label(label: str, labels: tuple[str, ...]) -> str:
    if not labels:
        return label
    try:
        index = labels.index(label)
    except ValueError:
        return labels[0]
    return labels[(index + 1) % len(labels)]
