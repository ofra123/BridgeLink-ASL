"""Training helpers for the baseline centroid model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .classifier import CentroidModel, save_model
from .dataset import DatasetRecord, load_dataset, summarize_splits, validate_dataset
from .vocabulary import get_v1_labels


@dataclass(frozen=True)
class TrainingSummary:
    dataset_path: str
    output_path: str
    labels_trained: tuple[str, ...]
    records_used: int
    split_distribution: dict[str, int]


def train_centroid_model(dataset_path: str | Path, output_path: str | Path) -> TrainingSummary:
    """Train and save a centroid baseline from the training split."""

    records = load_dataset(dataset_path)
    issues = validate_dataset(records, allowed_labels=get_v1_labels())
    if issues:
        raise ValueError("; ".join(issues))

    train_records = [record for record in records if record.split == "train"]
    if not train_records:
        raise ValueError("Dataset does not contain any training records.")

    grouped: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in train_records:
        grouped[record.label].append(record)

    centroids: dict[str, tuple[float, ...]] = {}
    sample_counts: dict[str, int] = {}
    for label, label_records in grouped.items():
        feature_count = len(label_records[0].landmarks)
        centroid = tuple(
            round(sum(record.landmarks[index] for record in label_records) / len(label_records), 6)
            for index in range(feature_count)
        )
        centroids[label] = centroid
        sample_counts[label] = len(label_records)

    model = CentroidModel(centroids=centroids, sample_counts=sample_counts, source="trained")
    saved_path = save_model(model, output_path)
    return TrainingSummary(
        dataset_path=str(Path(dataset_path).expanduser().resolve()),
        output_path=str(saved_path),
        labels_trained=tuple(sorted(centroids)),
        records_used=len(train_records),
        split_distribution=summarize_splits(records),
    )
