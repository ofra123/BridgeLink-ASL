"""Baseline centroid classifier and model I/O."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import LandmarkSample, Prediction
from .vocabulary import LANDMARK_FEATURE_LENGTH, get_seed_landmarks, get_v1_labels


@dataclass(frozen=True)
class CentroidModel:
    """Simple centroid classifier used for scaffolding and early baselines."""

    centroids: dict[str, tuple[float, ...]]
    sample_counts: dict[str, int]
    source: str = "seed"

    @property
    def feature_length(self) -> int:
        first_vector = next(iter(self.centroids.values()))
        return len(first_vector)

    def predict(self, sample: LandmarkSample) -> Prediction:
        if not sample.detected:
            return Prediction(label="NO_SIGN", confidence=0.0, frame_index=sample.frame_index, source=self.source)

        distances = sorted(
            (math.dist(sample.values, centroid), label)
            for label, centroid in self.centroids.items()
        )
        best_distance, best_label = distances[0]
        second_distance = distances[1][0] if len(distances) > 1 else best_distance + 1.0
        margin = max(second_distance - best_distance, 0.0)
        confidence = min(0.99, round((1.0 / (1.0 + best_distance)) * (1.0 + min(margin, 1.0) * 0.15), 4))
        return Prediction(label=best_label, confidence=confidence, frame_index=sample.frame_index, source=self.source)

    def as_dict(self) -> dict[str, object]:
        return {
            "model_type": "centroid-baseline",
            "feature_length": self.feature_length,
            "source": self.source,
            "labels": {
                label: {
                    "centroid": list(vector),
                    "samples": self.sample_counts.get(label, 0),
                }
                for label, vector in self.centroids.items()
            },
        }


def build_seed_model(labels: Iterable[str] | None = None) -> CentroidModel:
    """Build a deterministic classifier from the built-in seed landmarks."""

    selected_labels = tuple(labels or get_v1_labels())
    centroids = {label: get_seed_landmarks(label) for label in selected_labels}
    sample_counts = {label: 0 for label in selected_labels}
    return CentroidModel(centroids=centroids, sample_counts=sample_counts, source="seed")


def build_classifier(model_path: Path, labels: Iterable[str] | None = None) -> CentroidModel:
    """Load a saved model when it exists, otherwise use the seed classifier."""

    if model_path.exists():
        return load_model(model_path)
    return build_seed_model(labels)


def load_model(model_path: str | Path) -> CentroidModel:
    """Load a saved centroid model from disk."""

    path = Path(model_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    label_payload = payload.get("labels", {})
    centroids = {
        label: tuple(float(value) for value in metadata["centroid"])
        for label, metadata in label_payload.items()
    }
    sample_counts = {
        label: int(metadata.get("samples", 0))
        for label, metadata in label_payload.items()
    }
    feature_length = int(payload.get("feature_length", LANDMARK_FEATURE_LENGTH))
    if centroids and any(len(vector) != feature_length for vector in centroids.values()):
        raise ValueError("Model contains inconsistent centroid lengths.")
    return CentroidModel(centroids=centroids, sample_counts=sample_counts, source=str(path))


def save_model(model: CentroidModel, output_path: str | Path) -> Path:
    """Persist a centroid model to disk."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.as_dict(), indent=2), encoding="utf-8")
    return path
