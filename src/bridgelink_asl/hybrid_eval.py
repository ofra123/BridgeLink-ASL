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

CANDIDATE_TOP1_KEYS = (
    "cnn_top1",
    "model_top1",
    "candidate_top1",
    "transformer_top1",
)

CANDIDATE_TOP1_CONFIDENCE_KEYS = (
    "cnn_top1_confidence",
    "model_top1_confidence",
    "candidate_top1_confidence",
    "transformer_top1_confidence",
)

CANDIDATE_TOP5_KEYS = (
    "cnn_top5",
    "model_top5",
    "candidate_top5",
    "transformer_top5",
    "top5",
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


def candidate_model_name(row: dict[str, object]) -> str:
    """Infer which model generated the top-5 candidate list."""

    if row.get("candidate_model"):
        return str(row["candidate_model"])
    if row.get("cnn_top1") or row.get("cnn_top5"):
        return "cnn"
    if row.get("transformer_top1") or row.get("transformer_top5"):
        return "transformer"
    return "candidate_model"


def candidate_top1_label(row: dict[str, object]) -> str:
    """Extract the candidate generator's top-1 label."""

    for key in CANDIDATE_TOP1_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    labels = _top5_labels_without_top1_fallback(row)
    return labels[0] if labels else ""


def candidate_top1_confidence(row: dict[str, object]) -> object:
    """Extract the candidate generator's top-1 confidence when present."""

    for key in CANDIDATE_TOP1_CONFIDENCE_KEYS:
        value = row.get(key)
        if value is not None:
            return value
    return ""


def candidate_top5_labels(row: dict[str, object]) -> list[str]:
    """Extract top-5 candidate labels from a hybrid evaluation row."""

    labels = _top5_labels_without_top1_fallback(row)
    if not labels:
        top1 = candidate_top1_label(row)
        if top1:
            labels.append(top1)
    return labels


def _top5_labels_without_top1_fallback(row: dict[str, object]) -> list[str]:
    raw_top5: object = []
    for key in CANDIDATE_TOP5_KEYS:
        value = row.get(key)
        if value:
            raw_top5 = value
            break
    labels: list[str] = []
    if isinstance(raw_top5, list):
        for item in raw_top5:
            if isinstance(item, dict):
                label = item.get("label") or item.get("gloss") or item.get("prediction")
            else:
                label = item
            if label is not None:
                labels.append(str(label))
    return labels


def transformer_top5_labels(row: dict[str, object]) -> list[str]:
    """Backward-compatible alias for older tests and manifests."""

    return candidate_top5_labels(row)


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
    candidates = candidate_top5_labels(row)
    return (
        "You are classifying an isolated ASL sign from a short video. "
        f"Choose the best matching label from this candidate list only: {candidates}. "
        "Return only the chosen label and one short reason."
    )


def compute_hybrid_metrics(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Compute Transformer baseline and optional VLM reranking metrics."""

    materialized = list(rows)
    true_labels = [str(row.get("true_label", "")) for row in materialized]
    candidate_model = candidate_model_name(materialized[0]) if materialized else "candidate_model"
    candidate_top1 = [candidate_top1_label(row) for row in materialized]
    candidate_top5_hits = [
        normalize_label(actual) in {normalize_label(label) for label in candidate_top5_labels(row)}
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

    candidate_metrics = compute_classification_metrics(
        [normalize_label(label) for label in true_labels],
        [normalize_label(label) for label in candidate_top1],
    )

    result: dict[str, object] = {
        "num_samples": len(materialized),
        "num_classes": len({normalize_label(label) for label in true_labels if label}),
        "candidate_model": candidate_model,
        "candidate_top1_accuracy": candidate_metrics.accuracy,
        "candidate_top5_coverage": round(
            sum(candidate_top5_hits) / len(candidate_top5_hits), 4
        )
        if candidate_top5_hits
        else 0.0,
        "candidate_top1_metrics": candidate_metrics.as_dict(),
        "vlm_evaluated_samples": len(predicted_vlm),
    }
    # Backward-compatible names retained for existing report scripts.
    result["transformer_top1_accuracy"] = result["candidate_top1_accuracy"]
    result["transformer_top5_coverage"] = result["candidate_top5_coverage"]
    result["transformer_top1_metrics"] = result["candidate_top1_metrics"]
    if "cnn" in candidate_model.lower():
        result["cnn_top1_accuracy"] = result["candidate_top1_accuracy"]
        result["cnn_top5_coverage"] = result["candidate_top5_coverage"]
        result["cnn_top1_metrics"] = result["candidate_top1_metrics"]

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
        "candidate_model",
        "candidate_top1",
        "candidate_top1_confidence",
        "candidate_top5_labels",
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
                    "candidate_model": candidate_model_name(row),
                    "candidate_top1": candidate_top1_label(row),
                    "candidate_top1_confidence": candidate_top1_confidence(row),
                    "candidate_top5_labels": json.dumps(
                        candidate_top5_labels(row), ensure_ascii=False
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
