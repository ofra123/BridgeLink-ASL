"""Offline evaluation helpers for the baseline model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .classifier import build_classifier
from .dataset import load_dataset
from .types import LandmarkSample


@dataclass(frozen=True)
class EvaluationMetrics:
    split: str
    total_samples: int
    correct_samples: int
    accuracy: float
    model_source: str
    per_label_accuracy: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "accuracy": self.accuracy,
            "model_source": self.model_source,
            "per_label_accuracy": self.per_label_accuracy,
        }


def evaluate_saved_model(model_path: str | Path, dataset_path: str | Path, split: str = "test") -> EvaluationMetrics:
    """Evaluate a saved model against a dataset split."""

    records = [record for record in load_dataset(dataset_path) if record.split == split]
    if not records:
        raise ValueError(f"Dataset does not contain any records for split '{split}'.")

    classifier = build_classifier(Path(model_path).expanduser().resolve(), labels={record.label for record in records})

    correct = 0
    by_label_total: dict[str, int] = {}
    by_label_correct: dict[str, int] = {}
    for index, record in enumerate(records):
        prediction = classifier.predict(
            LandmarkSample(frame_index=index, values=record.landmarks, source="eval", detected=True)
        )
        by_label_total[record.label] = by_label_total.get(record.label, 0) + 1
        if prediction.label == record.label:
            correct += 1
            by_label_correct[record.label] = by_label_correct.get(record.label, 0) + 1

    per_label_accuracy = {
        label: round(by_label_correct.get(label, 0) / total, 4)
        for label, total in sorted(by_label_total.items())
    }
    accuracy = round(correct / len(records), 4)
    return EvaluationMetrics(
        split=split,
        total_samples=len(records),
        correct_samples=correct,
        accuracy=accuracy,
        model_source=classifier.source,
        per_label_accuracy=per_label_accuracy,
    )


def save_metrics(metrics: EvaluationMetrics, output_path: str | Path) -> Path:
    """Persist evaluation metrics to disk."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.as_dict(), indent=2), encoding="utf-8")
    return path
