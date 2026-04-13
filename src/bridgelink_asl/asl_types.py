"""Shared datatypes for BridgeLink ASL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Frame:
    """A single frame emitted by a frame source."""

    index: int
    source: str
    image: Any | None = None
    timestamp: float = 0.0


@dataclass(frozen=True)
class LandmarkSample:
    """A fixed-length landmark vector for one frame."""

    frame_index: int
    values: tuple[float, ...]
    source: str = "mock"
    detected: bool = True
    label_hint: str | None = None


@dataclass(frozen=True)
class Prediction:
    """A classifier prediction for a landmark sample."""

    label: str
    confidence: float
    frame_index: int
    source: str = "classifier"


@dataclass(frozen=True)
class TranslationEvent:
    """A stable user-facing translation event."""

    label: str
    text: str
    confidence: float
    frame_index: int
    tts_provider: str


@dataclass(frozen=True)
class RunSummary:
    """Summary information from a demo run."""

    frames_processed: int
    frame_source: str
    classifier_source: str
    tts_provider: str
    model_path: str
    events: tuple[TranslationEvent, ...] = field(default_factory=tuple)
