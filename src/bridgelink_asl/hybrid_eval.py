"""Utilities for evaluating Transformer + VLM hybrid reranking."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from bridgelink_asl.metrics import compute_classification_metrics

PREDICTION_KEYS = (
    "vlm_prediction",
    "vlm_label",
    "vlm_selected_label",
    "vlm_top1",
    "prediction",
)


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load newline-delimited JSON rows."""

    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} must be a JSON object.")
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    """Write newline-delimited JSON rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(label: object) -> str:
    """Normalize labels for robust matching across VLM text outputs."""

    return str(label or "").strip().lower().replace("_", " ")


def transformer_top5_labels(row: dict[str, object]) -> list[str]:
    """Extract top-5 candidate labels from a hybrid evaluation row."""

    raw_top5 = row.get("transformer_top5") or row.get("top5") or []
    labels: list[str] = []
    if isinstance(raw_top5, list):
        for item in raw_top5:
            if isinstance(item, dict):
                label = item.get("label") or item.get("gloss") or item.get("prediction")
            else:
                label = item
            if label is not None:
                labels.append(str(label))
    if not labels and row.get("transformer_top1"):
        labels.append(str(row["transformer_top1"]))
    return labels


def get_vlm_prediction(row: dict[str, object]) -> str | None:
    """Return the first available VLM prediction from a row."""

    for key in PREDICTION_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def build_vlm_prompt(row: dict[str, object]) -> str:
    """Build a constrained reranking prompt for VLM review."""

    if row.get("vlm_prompt"):
        return str(row["vlm_prompt"])
    candidates = transformer_top5_labels(row)
    return (
        "You are classifying an isolated ASL sign from a short video. "
        f"Choose the best matching label from this candidate list only: {candidates}. "
        "Return only the chosen label and one short reason."
    )


def compute_hybrid_metrics(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Compute Transformer baseline and optional VLM reranking metrics."""

    materialized = list(rows)
    true_labels = [str(row.get("true_label", "")) for row in materialized]
    transformer_top1 = [str(row.get("transformer_top1", "")) for row in materialized]
    transformer_top5_hits = [
        normalize_label(actual) in {normalize_label(label) for label in transformer_top5_labels(row)}
        for actual, row in zip(true_labels, materialized)
    ]

    predicted_vlm: list[str] = []
    expected_vlm: list[str] = []
    for row in materialized:
        prediction = get_vlm_prediction(row)
        if prediction is None:
            continue
        expected_vlm.append(str(row.get("true_label", "")))
        predicted_vlm.append(prediction)

    transformer_metrics = compute_classification_metrics(
        [normalize_label(label) for label in true_labels],
        [normalize_label(label) for label in transformer_top1],
    )

    result: dict[str, object] = {
        "num_samples": len(materialized),
        "num_classes": len({normalize_label(label) for label in true_labels if label}),
        "transformer_top1_accuracy": transformer_metrics.accuracy,
        "transformer_top5_coverage": round(
            sum(transformer_top5_hits) / len(transformer_top5_hits), 4
        )
        if transformer_top5_hits
        else 0.0,
        "transformer_top1_metrics": transformer_metrics.as_dict(),
        "vlm_evaluated_samples": len(predicted_vlm),
    }

    if predicted_vlm:
        vlm_metrics = compute_classification_metrics(
            [normalize_label(label) for label in expected_vlm],
            [normalize_label(label) for label in predicted_vlm],
        )
        result["vlm_rerank_accuracy"] = vlm_metrics.accuracy
        result["vlm_rerank_metrics"] = vlm_metrics.as_dict()
    else:
        result["vlm_rerank_accuracy"] = None
        result["vlm_rerank_metrics"] = None

    return result


def write_review_csv(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    """Write a CSV template for manual or VLM-assisted reranking."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "true_label",
        "video_path",
        "transformer_top1",
        "transformer_top1_confidence",
        "transformer_top5_labels",
        "vlm_prompt",
        "vlm_prediction",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": row.get("video_id", ""),
                    "true_label": row.get("true_label", ""),
                    "video_path": row.get("video_path", ""),
                    "transformer_top1": row.get("transformer_top1", ""),
                    "transformer_top1_confidence": row.get(
                        "transformer_top1_confidence", ""
                    ),
                    "transformer_top5_labels": json.dumps(
                        transformer_top5_labels(row), ensure_ascii=False
                    ),
                    "vlm_prompt": build_vlm_prompt(row),
                    "vlm_prediction": get_vlm_prediction(row) or "",
                    "notes": row.get("notes", ""),
                }
            )


def merge_predictions(
    rows: Iterable[dict[str, object]], predictions_path: str | Path
) -> list[dict[str, object]]:
    """Merge VLM predictions from a CSV or JSONL file into eval rows."""

    prediction_map = _load_prediction_map(predictions_path)
    merged: list[dict[str, object]] = []
    for row in rows:
        updated = dict(row)
        key = _row_key(row)
        prediction = prediction_map.get(key)
        if prediction:
            updated["vlm_prediction"] = prediction
        merged.append(updated)
    return merged


def _load_prediction_map(path: str | Path) -> dict[str, str]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        with input_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return {
                _row_key(row): prediction
                for row in reader
                if (prediction := _extract_prediction_from_mapping(row))
            }

    rows = load_jsonl(input_path)
    return {
        _row_key(row): prediction
        for row in rows
        if (prediction := _extract_prediction_from_mapping(row))
    }


def _extract_prediction_from_mapping(row: dict[str, object]) -> str | None:
    for key in PREDICTION_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _row_key(row: dict[str, object]) -> str:
    video_id = str(row.get("video_id") or "").strip()
    if video_id:
        return video_id
    return str(row.get("video_path") or row.get("clip_path") or "").strip()
