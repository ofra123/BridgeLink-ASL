from __future__ import annotations

import csv
import json

from bridgelink_asl.hybrid_eval import (
    compute_hybrid_metrics,
    load_jsonl,
    merge_predictions,
    transformer_top5_labels,
    write_jsonl,
    write_review_csv,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "video_id": "a1",
            "true_label": "basketball",
            "video_path": "clips/a1.mp4",
            "transformer_top1": "ball",
            "transformer_top5": [
                {"label": "ball", "confidence": 0.4},
                {"label": "basketball", "confidence": 0.3},
            ],
        },
        {
            "video_id": "b2",
            "true_label": "change",
            "video_path": "clips/b2.mp4",
            "transformer_top1": "change",
            "transformer_top5": [
                {"label": "change", "confidence": 0.7},
                {"label": "argue", "confidence": 0.1},
            ],
        },
        {
            "video_id": "c3",
            "true_label": "doctor",
            "video_path": "clips/c3.mp4",
            "transformer_top1": "before",
            "transformer_top5": [{"label": "before", "confidence": 0.5}],
        },
    ]


def test_transformer_top5_labels_handles_dict_candidates() -> None:
    assert transformer_top5_labels(_rows()[0]) == ["ball", "basketball"]


def test_compute_hybrid_metrics_scores_baseline_and_vlm() -> None:
    rows = _rows()
    rows[0]["vlm_prediction"] = "basketball"
    rows[1]["vlm_prediction"] = "change"
    rows[2]["vlm_prediction"] = "before"

    metrics = compute_hybrid_metrics(rows)

    assert metrics["num_samples"] == 3
    assert metrics["num_classes"] == 3
    assert metrics["transformer_top1_accuracy"] == 0.3333
    assert metrics["transformer_top5_coverage"] == 0.6667
    assert metrics["vlm_evaluated_samples"] == 3
    assert metrics["vlm_rerank_accuracy"] == 0.6667


def test_write_review_csv_and_merge_predictions(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    review_csv = tmp_path / "review.csv"
    write_jsonl(_rows(), manifest)
    rows = load_jsonl(manifest)
    write_review_csv(rows, review_csv)

    with review_csv.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    review_rows[0]["vlm_prediction"] = "basketball"
    review_rows[1]["vlm_prediction"] = "argue"

    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_rows[0].keys())
        writer.writeheader()
        writer.writerows(review_rows)

    merged = merge_predictions(rows, predictions)

    assert merged[0]["vlm_prediction"] == "basketball"
    assert merged[1]["vlm_prediction"] == "argue"
    assert "vlm_prediction" not in merged[2]


def test_load_jsonl_rejects_non_object_rows(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")

    try:
        load_jsonl(path)
    except ValueError as exc:
        assert "must be a JSON object" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
