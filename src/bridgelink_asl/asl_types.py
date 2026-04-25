"""Shared datatypes for BridgeLink ASL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
class DetectedGestureToken:
    """One stable token detected within a sentence-sized gesture window."""

    label: str
    confidence: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class GestureWindow:
    """Input contract shared by CNN, VLM, and compare wrapper modes."""

    clip_id: str
    sampled_frames: tuple[Path, ...]
    token_trace: tuple[DetectedGestureToken, ...]
    expected_gloss: tuple[str, ...] = field(default_factory=tuple)
    expected_text: str = ""
    source: str = "manifest"


@dataclass(frozen=True)
class SentenceEvent:
    """Sentence-level output from the CNN or VLM path."""

    gloss: tuple[str, ...]
    sentence: str
    confidence: float
    model_mode: str
    needs_clarification: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    """Per-clip comparison output written by the wrapper."""

    clip_id: str
    expected_text: str
    token_trace: tuple[DetectedGestureToken, ...]
    cnn_prediction: SentenceEvent | None = None
    vlm_prediction: SentenceEvent | None = None
    cnn_latency_ms: float | None = None
    vlm_latency_ms: float | None = None
    failure_notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RunSummary:
    """Summary information from a demo run."""

    frames_processed: int
    frame_source: str
    classifier_source: str
    tts_provider: str
    model_path: str
    events: tuple[TranslationEvent, ...] = field(default_factory=tuple)
