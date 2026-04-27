"""Sentence wrapper modes for CNN, VLM, and side-by-side comparison."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import cv2

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


@dataclass
class LocalQwen25VlmInterpreter:
    """Local Qwen2.5-VL interpreter with lazy optional backend loading."""

    model_id: str
    max_visuals: int = 6
    max_new_tokens: int = 160
    temperature: float = 0.0
    _processor: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)

    def interpret(self, window: GestureWindow) -> SentenceEvent:
        prompt = self._build_prompt(window)
        with self._build_messages(window, prompt) as messages:
            response_text = self._generate_response_text(messages)
            return self._parse_response(response_text, window)

    def _build_prompt(self, window: GestureWindow) -> str:
        token_lines = [
            f"- {token.label} (confidence={token.confidence:.2f}, frames={token.start_frame}-{token.end_frame})"
            for token in window.token_trace
        ]
        candidate_lines = [f"- {label}" for label in window.candidate_labels]

        instructions = [
            "You are analyzing an American Sign Language clip.",
            "Use the visual evidence first, then use the token trace as a weak hint.",
            "Return JSON only.",
        ]
        if candidate_lines:
            instructions.extend(
                [
                    "Choose exactly one label from the candidate list.",
                    'Return JSON with keys: "gloss", "sentence", "confidence", and "needs_clarification".',
                    'The "gloss" value must be a one-item list containing the chosen candidate label in uppercase.',
                ]
            )
        else:
            instructions.extend(
                [
                    'Return JSON with keys: "gloss", "sentence", "confidence", and "needs_clarification".',
                    'Use "gloss" for the ASL gloss tokens and "sentence" for the readable English sentence.',
                ]
            )

        sections = ["\n".join(instructions)]
        if candidate_lines:
            sections.append("Allowed candidate labels:\n" + "\n".join(candidate_lines))
        if token_lines:
            sections.append("Token trace hint:\n" + "\n".join(token_lines))
        sections.append("Respond with JSON only and no markdown fences.")
        return "\n\n".join(sections)

    @contextmanager
    def _build_messages(self, window: GestureWindow, prompt: str):
        frame_paths = [path.resolve() for path in window.sampled_frames if path.exists()]
        if frame_paths:
            frame_paths = _uniform_subset(frame_paths, self.max_visuals)
            yield [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": [str(path) for path in frame_paths]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return

        if window.video_path and window.video_path.exists():
            with TemporaryDirectory(prefix="bridgelink-qwen-frames-") as temp_dir:
                frame_paths = self._extract_video_frames(window.video_path, Path(temp_dir))
                yield [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": [str(path) for path in frame_paths]},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            return

        yield [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def _extract_video_frames(self, video_path: Path, output_dir: Path) -> list[Path]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video clip: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.max_visuals
        indices = _uniform_indices(total_frames, self.max_visuals)

        saved_paths: list[Path] = []
        for output_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            path = output_dir / f"frame_{output_index:02d}.png"
            cv2.imwrite(str(path), frame)
            saved_paths.append(path)

        capture.release()
        if not saved_paths:
            raise RuntimeError(f"Could not decode frames from video clip: {video_path}")
        return saved_paths

    def _generate_response_text(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        processor, model = self._ensure_backend()
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prepared_inputs = self._move_inputs_to_model(inputs, model)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generation_kwargs["temperature"] = self.temperature

        generated_ids = model.generate(**prepared_inputs, **generation_kwargs)
        input_ids = prepared_inputs.get("input_ids")
        if input_ids is not None and hasattr(generated_ids, "__getitem__"):
            generated_ids = generated_ids[:, input_ids.shape[-1] :]

        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            raise RuntimeError("Local Qwen2.5-VL returned no text.")
        return str(decoded[0]).strip()

    def _ensure_backend(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Install the VLM extras with "
                '`pip install -e ".[vlm]"` to enable local Qwen2.5-VL inference.'
            ) from exc

        model_class = None
        try:
            from transformers import AutoModelForImageTextToText as model_class  # type: ignore[attr-defined]
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as model_class  # type: ignore[attr-defined]
            except ImportError:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration as model_class  # type: ignore[attr-defined]
                except ImportError as exc:
                    raise RuntimeError(
                        "Your transformers build does not expose a vision-language model loader "
                        "compatible with Qwen2.5-VL."
                    ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = model_class.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        return self._processor, self._model

    def _move_inputs_to_model(self, inputs: Any, model: Any) -> dict[str, Any]:
        if not hasattr(inputs, "items"):
            return inputs

        target_device = getattr(model, "device", None)
        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            if target_device is not None and hasattr(value, "to"):
                prepared[key] = value.to(target_device)
            else:
                prepared[key] = value
        return prepared

    def _parse_response(self, response_text: str, window: GestureWindow) -> SentenceEvent:
        payload = _extract_json_payload(response_text)
        if payload is not None:
            return self._sentence_event_from_payload(payload, window, response_text)
        return self._sentence_event_from_free_text(response_text, window)

    def _sentence_event_from_payload(
        self,
        payload: dict[str, Any],
        window: GestureWindow,
        raw_text: str,
    ) -> SentenceEvent:
        gloss = _coerce_gloss_tokens(payload.get("gloss"))
        sentence = str(payload.get("sentence", "")).strip()
        confidence = _coerce_confidence(payload.get("confidence"), fallback=_average_confidence(window.token_trace))
        needs_clarification = bool(payload.get("needs_clarification", False))
        failure_reason = _optional_text(payload.get("reason")) or _optional_text(payload.get("failure_reason"))

        chosen_label = _resolve_chosen_label(
            gloss=gloss,
            sentence=sentence,
            response_text=raw_text,
            candidates=window.candidate_labels,
        )
        if window.candidate_labels:
            if chosen_label is None:
                raise RuntimeError("Local Qwen response did not choose one of the allowed candidate labels.")
            gloss = (chosen_label,)
            sentence = sentence or _pretty_sentence_from_label(chosen_label)
        elif not gloss:
            gloss = tuple(token.label for token in window.token_trace)

        if not sentence:
            sentence = _natural_sentence_from_gloss(gloss)

        return SentenceEvent(
            gloss=gloss,
            sentence=sentence,
            confidence=confidence,
            model_mode="vlm",
            needs_clarification=needs_clarification,
            failure_reason=failure_reason,
        )

    def _sentence_event_from_free_text(self, response_text: str, window: GestureWindow) -> SentenceEvent:
        chosen_label = _resolve_chosen_label(
            gloss=(),
            sentence="",
            response_text=response_text,
            candidates=window.candidate_labels,
        )
        if chosen_label is None:
            raise RuntimeError("Local Qwen response was not valid JSON and no candidate label could be recovered.")

        reason = _optional_text(_extract_reason(response_text))
        return SentenceEvent(
            gloss=(chosen_label,),
            sentence=_pretty_sentence_from_label(chosen_label),
            confidence=_average_confidence(window.token_trace),
            model_mode="vlm",
            needs_clarification=False,
            failure_reason=reason,
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
    if record.gloss:
        tokens = tuple(
            DetectedGestureToken(
                label=label,
                confidence=round(max(0.55, 0.93 - index * 0.07), 2),
                start_frame=index * 4,
                end_frame=index * 4 + 3,
            )
            for index, label in enumerate(record.gloss)
        )
    else:
        tokens = ()

    return GestureWindow(
        clip_id=record.clip_id,
        sampled_frames=record.sampled_frames,
        token_trace=tokens,
        video_path=record.video_path,
        candidate_labels=record.candidate_labels,
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


def _extract_json_payload(response_text: str) -> dict[str, Any] | None:
    stripped = response_text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        candidates.insert(0, stripped[start:end])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _coerce_gloss_tokens(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(token).strip().upper() for token in value if str(token).strip())


def _coerce_confidence(value: Any, *, fallback: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = fallback
    return max(0.0, min(1.0, confidence))


def _resolve_chosen_label(
    *,
    gloss: tuple[str, ...],
    sentence: str,
    response_text: str,
    candidates: tuple[str, ...],
) -> str | None:
    if not candidates:
        return gloss[0] if gloss else None

    normalized = {_normalize_label(label): label for label in candidates}
    search_order: list[str] = []
    search_order.extend(gloss)
    if sentence:
        search_order.append(sentence)
    search_order.append(response_text)

    for item in search_order:
        match = _match_candidate(item, normalized)
        if match:
            return match
    return None


def _match_candidate(text: str, normalized_candidates: dict[str, str]) -> str | None:
    normalized_text = _normalize_label(text)
    if normalized_text in normalized_candidates:
        return normalized_candidates[normalized_text]

    for normalized_label, original in normalized_candidates.items():
        pattern = rf"(?<![A-Z0-9]){re.escape(normalized_label)}(?![A-Z0-9])"
        if re.search(pattern, normalized_text):
            return original
    return None


def _normalize_label(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_ ]+", " ", str(text).strip().upper())
    return re.sub(r"\s+", " ", cleaned).strip()


def _pretty_sentence_from_label(label: str) -> str:
    text = label.replace("_", " ").lower()
    return text[:1].upper() + text[1:] + "."


def _extract_reason(response_text: str) -> str | None:
    stripped = response_text.strip()
    if ":" in stripped:
        _, reason = stripped.split(":", 1)
        return reason.strip()
    if "-" in stripped:
        _, reason = stripped.split("-", 1)
        return reason.strip()
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _uniform_subset(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    indices = _uniform_indices(len(paths), max_items)
    return [paths[index] for index in indices]


def _uniform_indices(total_items: int, max_items: int) -> list[int]:
    if total_items <= 0 or max_items <= 0:
        return []
    if total_items <= max_items:
        return list(range(total_items))
    if max_items == 1:
        return [0]

    step = (total_items - 1) / (max_items - 1)
    return [min(total_items - 1, round(index * step)) for index in range(max_items)]
