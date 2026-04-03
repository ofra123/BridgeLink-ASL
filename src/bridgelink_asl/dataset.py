"""Dataset loading and validation helpers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetRecord:
    label: str
    split: str
    landmarks: tuple[float, ...]


def load_dataset(dataset_path: str | Path) -> list[DatasetRecord]:
    """Load a JSONL landmark dataset from disk."""

    path = Path(dataset_path).expanduser().resolve()
    records: list[DatasetRecord] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        records.append(
            DatasetRecord(
                label=str(payload["label"]).strip().upper(),
                split=str(payload["split"]).strip().lower(),
                landmarks=tuple(float(value) for value in payload["landmarks"]),
            )
        )
    return records


def validate_dataset(records: Iterable[DatasetRecord], allowed_labels: Iterable[str] | None = None) -> list[str]:
    """Return human-readable validation issues for a dataset."""

    materialized = list(records)
    issues: list[str] = []
    if not materialized:
        return ["Dataset is empty."]

    feature_lengths = {len(record.landmarks) for record in materialized}
    if len(feature_lengths) != 1:
        issues.append("Dataset contains inconsistent landmark vector lengths.")

    allowed = {label.upper() for label in allowed_labels} if allowed_labels else None
    unknown_labels = sorted({record.label for record in materialized if allowed and record.label not in allowed})
    if unknown_labels:
        issues.append(f"Dataset contains unknown labels: {', '.join(unknown_labels)}.")

    bad_splits = sorted({record.split for record in materialized if record.split not in {"train", "val", "test"}})
    if bad_splits:
        issues.append(f"Dataset contains unsupported splits: {', '.join(bad_splits)}.")

    return issues


def summarize_splits(records: Iterable[DatasetRecord]) -> dict[str, int]:
    """Count how many records appear in each split."""

    return dict(Counter(record.split for record in records))
