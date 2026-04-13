"""End-to-end demo session orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .camera import build_frame_source
from .classifier import build_classifier
from .config import AppConfig
from .landmarks import build_landmark_extractor
from .smoothing import PredictionSmoother
from .speech import SpeechAdapterSelection, select_speech_adapter
from .translation import SignTranslator
from .asl_types import RunSummary, TranslationEvent


class JsonlTranscriptWriter:
    """Append translation events to a JSONL transcript file."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def append(self, event: TranslationEvent) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frame_index": event.frame_index,
            "label": event.label,
            "text": event.text,
            "confidence": event.confidence,
            "tts_provider": event.tts_provider,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


@dataclass
class BridgeLinkSession:
    """A runnable demo session built from config."""

    config: AppConfig
    frame_source: object
    landmark_extractor: object
    classifier: object
    translator: SignTranslator
    smoother: PredictionSmoother
    speech_selection: SpeechAdapterSelection
    transcript_writer: JsonlTranscriptWriter

    def run(self) -> RunSummary:
        events: list[TranslationEvent] = []
        frames_processed = 0

        for frame in self.frame_source.frames(self.config.max_frames):
            frames_processed += 1
            landmark_sample = self.landmark_extractor.extract(frame)
            prediction = self.classifier.predict(landmark_sample)
            stable_prediction = self.smoother.observe(prediction)
            if stable_prediction is None:
                continue

            event = self.translator.to_event(stable_prediction, self.speech_selection.resolved_provider)
            self.speech_selection.adapter.speak(event.text)
            self.transcript_writer.append(event)
            events.append(event)

        return RunSummary(
            frames_processed=frames_processed,
            frame_source=self.frame_source.name,
            classifier_source=self.classifier.source,
            tts_provider=self.speech_selection.resolved_provider,
            model_path=str(self.config.model_path),
            events=tuple(events),
        )


def build_session(config: AppConfig) -> BridgeLinkSession:
    """Construct the full demo session from config."""

    return BridgeLinkSession(
        config=config,
        frame_source=build_frame_source(config),
        landmark_extractor=build_landmark_extractor(config),
        classifier=build_classifier(config.model_path, labels=config.demo_sequence),
        translator=SignTranslator(),
        smoother=PredictionSmoother(
            hold_frames=config.hold_frames,
            confidence_threshold=config.confidence_threshold,
        ),
        speech_selection=select_speech_adapter(config),
        transcript_writer=JsonlTranscriptWriter(config.transcript_path),
    )
