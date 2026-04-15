"""Classification metrics used by CNN and VLM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ClassificationMetrics:
    labels: tuple[str, ...]
    total: int
    correct: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: dict[str, dict[str, float | int]]
    confusion_matrix: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_label": self.per_label,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_classification_metrics(
    expected: Iterable[str],
    predicted: Iterable[str],
    labels: Iterable[str] | None = None,
) -> ClassificationMetrics:
    """Compute accuracy, macro precision, recall, F1, and confusion counts."""

    expected_list = [str(label) for label in expected]
    predicted_list = [str(label) for label in predicted]
    if len(expected_list) != len(predicted_list):
        raise ValueError("Expected and predicted label lists must have the same length.")

    label_set = set(labels or ())
    label_set.update(expected_list)
    label_set.update(predicted_list)
    ordered_labels = tuple(sorted(label_set))
    confusion = {actual: {label: 0 for label in ordered_labels} for actual in ordered_labels}

    correct = 0
    for actual, guess in zip(expected_list, predicted_list):
        confusion.setdefault(actual, {label: 0 for label in ordered_labels})
        confusion[actual][guess] = confusion[actual].get(guess, 0) + 1
        if actual == guess:
            correct += 1

    per_label: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in ordered_labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(actual, {}).get(label, 0) for actual in ordered_labels if actual != label)
        fn = sum(count for guess, count in confusion.get(label, {}).items() if guess != label)
        support = sum(confusion.get(label, {}).values())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_label[label] = {
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    total = len(expected_list)
    return ClassificationMetrics(
        labels=ordered_labels,
        total=total,
        correct=correct,
        accuracy=round(_safe_div(correct, total), 4),
        macro_precision=round(_mean(precisions), 4),
        macro_recall=round(_mean(recalls), 4),
        macro_f1=round(_mean(f1_scores), 4),
        per_label=per_label,
        confusion_matrix=confusion,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
