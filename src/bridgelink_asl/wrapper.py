"""Sentence wrapper modes for CNN, VLM, and side-by-side comparison."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .asl_types import ComparisonResult, DetectedGestureToken, GestureWindow, SentenceEvent
from .clip_dataset import ClipDatasetRecord, load_clip_dataset, validate_clip_dataset
from .translation import SignTranslator


class SentenceInterpreter(Protocol):
    """Interface for sentence-level interpretation over a gesture window."""

    def interpret(self, window: GestureWindow) -> SentenceEvent:
        """Return a sentence-level prediction for one clip-sized window."""


@dataclass(frozen=True)
class MockSentenceInterpreter:
    """Offline-safe deterministic VLM substitute for tests and demos."""

    confidence: float = 0.84

    def interpret(self, window: GestureWindow) -> SentenceEvent:
        gloss = tuple(token.label for token in window.token_trace)
        sentence = _natural_sentence_from_gloss(gloss)
        return SentenceEvent(
            gloss=gloss,
            sentence=sentence,
            confidence=self.confidence,
            model_mode="vlm",
            needs_clarification=False,
        )


@dataclass(frozen=True)
class LocalQwen25VlmInterpreter:
    """Placeholder contract for the eventual local Qwen provider."""

    model_id: str

    def interpret(self, window: GestureWindow) -> SentenceEvent:
        raise RuntimeError(
            "Local Qwen2.5-VL inference is not wired in yet. "
            "Use the mock interpreter for offline demos and tests."
        )


@dataclass(frozen=True)
class WrapperSummary:
    """High-level outcome from a wrapper run."""

    mode: str
    manifest_path: str
    output_path: str
    records_processed: int
    failures: int


class JsonlComparisonWriter:
    """Append wrapper results to a JSONL output file."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path).expanduser().resolve()

    def write(self, result: ComparisonResult) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_comparison_result_to_payload(result)) + "\n")


def run_wrapper(
    manifest_path: str | Path,
    *,
    mode: str,
    output_path: str | Path = "outputs/comparison-results.jsonl",
    interpreter: SentenceInterpreter | None = None,
    vlm_confidence_floor: float = 0.6,
) -> WrapperSummary:
    """Run one of the documented phase-3 wrapper modes over a clip manifest."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"cnn", "vlm", "compare"}:
        raise ValueError("Wrapper mode must be one of: cnn, vlm, compare.")

    manifest = Path(manifest_path).expanduser().resolve()
    records = load_clip_dataset(manifest)
    issues = validate_clip_dataset(records)
    if issues:
        raise ValueError("; ".join(issues))

    writer = JsonlComparisonWriter(output_path)
    active_interpreter = interpreter or MockSentenceInterpreter()
    failures = 0

    for record in records:
        result = _run_record(
            record,
            mode=normalized_mode,
            interpreter=active_interpreter,
            vlm_confidence_floor=vlm_confidence_floor,
        )
        writer.write(result)
        if result.failure_notes:
            failures += 1

    return WrapperSummary(
        mode=normalized_mode,
        manifest_path=str(manifest),
        output_path=str(Path(output_path).expanduser().resolve()),
        records_processed=len(records),
        failures=failures,
    )


def _run_record(
    record: ClipDatasetRecord,
    *,
    mode: str,
    interpreter: SentenceInterpreter,
    vlm_confidence_floor: float,
) -> ComparisonResult:
    window = _build_gesture_window(record)
    notes: list[str] = []

    cnn_prediction: SentenceEvent | None = None
    cnn_latency_ms: float | None = None
    if mode in {"cnn", "compare"}:
        cnn_start = time.perf_counter()
        cnn_prediction = _predict_with_cnn_baseline(window)
        cnn_latency_ms = round((time.perf_counter() - cnn_start) * 1000.0, 3)

    vlm_prediction: SentenceEvent | None = None
    vlm_latency_ms: float | None = None
    if mode in {"vlm", "compare"}:
        vlm_start = time.perf_counter()
        try:
            candidate = interpreter.interpret(window)
        except Exception as exc:
            notes.append(f"VLM interpreter failed: {exc}")
            candidate = _fallback_sentence_event(window, model_mode="vlm", failure_reason=str(exc))

        if not candidate.sentence.strip() or candidate.confidence < vlm_confidence_floor:
            notes.append("VLM confidence below threshold; used gloss fallback.")
            fallback_reason = candidate.failure_reason or "low confidence"
            candidate = _fallback_sentence_event(window, model_mode="vlm", failure_reason=fallback_reason)

        vlm_prediction = candidate
        vlm_latency_ms = round((time.perf_counter() - vlm_start) * 1000.0, 3)

    return ComparisonResult(
        clip_id=record.clip_id,
        expected_text=record.english,
        token_trace=window.token_trace,
        cnn_prediction=cnn_prediction,
        vlm_prediction=vlm_prediction,
        cnn_latency_ms=cnn_latency_ms,
        vlm_latency_ms=vlm_latency_ms,
        failure_notes=tuple(notes),
    )


def _build_gesture_window(record: ClipDatasetRecord) -> GestureWindow:
    tokens = tuple(
        DetectedGestureToken(
            label=label,
            confidence=round(max(0.55, 0.93 - index * 0.07), 2),
            start_frame=index * 4,
            end_frame=index * 4 + 3,
        )
        for index, label in enumerate(record.gloss)
    )
    return GestureWindow(
        clip_id=record.clip_id,
        sampled_frames=record.sampled_frames,
        token_trace=tokens,
        expected_gloss=record.gloss,
        expected_text=record.english,
        source=record.source,
    )


def _predict_with_cnn_baseline(window: GestureWindow) -> SentenceEvent:
    gloss = tuple(token.label for token in window.token_trace)
    return SentenceEvent(
        gloss=gloss,
        sentence=_gloss_fallback_sentence(gloss),
        confidence=_average_confidence(window.token_trace),
        model_mode="cnn",
        needs_clarification=False,
    )


def _fallback_sentence_event(
    window: GestureWindow,
    *,
    model_mode: str,
    failure_reason: str,
) -> SentenceEvent:
    gloss = tuple(token.label for token in window.token_trace)
    return SentenceEvent(
        gloss=gloss,
        sentence=_gloss_fallback_sentence(gloss),
        confidence=_average_confidence(window.token_trace),
        model_mode=model_mode,
        needs_clarification=True,
        failure_reason=failure_reason,
    )


def _average_confidence(tokens: tuple[DetectedGestureToken, ...]) -> float:
    if not tokens:
        return 0.0
    return round(sum(token.confidence for token in tokens) / len(tokens), 3)


def _gloss_fallback_sentence(gloss: tuple[str, ...]) -> str:
    translator = SignTranslator()
    words = [translator.to_text(token).lower() for token in gloss]
    if not words:
        return ""
    sentence = " ".join(words)
    return sentence[:1].upper() + sentence[1:] + "."


def _natural_sentence_from_gloss(gloss: tuple[str, ...]) -> str:
    known_sequences = {
        ("HELLO", "WANT", "DRINK"): "Hello, I want a drink.",
        ("PLEASE", "HELP"): "Please help.",
        ("THANK_YOU", "FINISHED"): "Thank you, I am finished.",
        ("NO", "STOP"): "No, stop.",
    }
    return known_sequences.get(gloss, _gloss_fallback_sentence(gloss))


def _comparison_result_to_payload(result: ComparisonResult) -> dict[str, object]:
    return {
        "clip_id": result.clip_id,
        "expected_text": result.expected_text,
        "token_trace": [
            {
                "label": token.label,
                "confidence": token.confidence,
                "start_frame": token.start_frame,
                "end_frame": token.end_frame,
            }
            for token in result.token_trace
        ],
        "cnn_prediction": _sentence_event_to_payload(result.cnn_prediction),
        "vlm_prediction": _sentence_event_to_payload(result.vlm_prediction),
        "cnn_latency_ms": result.cnn_latency_ms,
        "vlm_latency_ms": result.vlm_latency_ms,
        "failure_notes": list(result.failure_notes),
    }


def _sentence_event_to_payload(event: SentenceEvent | None) -> dict[str, object] | None:
    if event is None:
        return None
    return {
        "gloss": list(event.gloss),
        "sentence": event.sentence,
        "confidence": event.confidence,
        "model_mode": event.model_mode,
        "needs_clarification": event.needs_clarification,
        "failure_reason": event.failure_reason,
    }
