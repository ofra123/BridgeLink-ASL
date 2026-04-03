"""Prediction smoothing helpers for real-time demo stability."""

from __future__ import annotations

from .types import Prediction


class PredictionSmoother:
    """Emit only stable predictions that survive several frames."""

    def __init__(self, hold_frames: int, confidence_threshold: float) -> None:
        self.hold_frames = max(1, hold_frames)
        self.confidence_threshold = confidence_threshold
        self._candidate_label: str | None = None
        self._candidate_count = 0
        self._last_emitted_label: str | None = None

    def observe(self, prediction: Prediction) -> Prediction | None:
        if prediction.confidence < self.confidence_threshold:
            self._candidate_label = None
            self._candidate_count = 0
            return None

        if prediction.label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = prediction.label
            self._candidate_count = 1

        if self._candidate_count < self.hold_frames:
            return None
        if prediction.label == self._last_emitted_label:
            return None

        self._last_emitted_label = prediction.label
        return prediction
