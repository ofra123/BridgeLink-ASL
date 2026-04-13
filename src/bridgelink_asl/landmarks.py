"""Landmark extraction boundaries for the starter demo."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .asl_types import Frame, LandmarkSample
from .vocabulary import DEFAULT_DEMO_SEQUENCE, get_seed_landmarks


@dataclass
class MockHandLandmarkExtractor:
    """Generate stable seed-aligned landmark vectors for the scaffold."""

    demo_sequence: tuple[str, ...] = DEFAULT_DEMO_SEQUENCE
    repeat_window: int = 4

    def extract(self, frame: Frame) -> LandmarkSample:
        label = self.demo_sequence[(frame.index // self.repeat_window) % len(self.demo_sequence)]
        base_vector = get_seed_landmarks(label)
        offset = ((frame.index % self.repeat_window) - (self.repeat_window / 2)) * 0.0025
        values = tuple(round(min(1.0, max(0.0, value + offset)), 4) for value in base_vector)
        return LandmarkSample(
            frame_index=frame.index,
            values=values,
            source="mock",
            detected=True,
            label_hint=label,
        )


def build_landmark_extractor(config: AppConfig) -> MockHandLandmarkExtractor:
    """Build the phase-1 landmark extractor."""

    return MockHandLandmarkExtractor(
        demo_sequence=config.demo_sequence,
        repeat_window=max(2, config.hold_frames + 1),
    )
