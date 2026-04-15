"""Hosted Hugging Face Space proof-of-concept inference.

This module is intentionally lightweight. It gives the Space a complete
end-to-end demo without requiring trained model artifacts or a paid 32B VLM
runtime on day one.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


POC_SENTENCES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("HELLO", "WANT", "DRINK"), "Hello, I want a drink."),
    (("PLEASE", "HELP"), "Please help."),
    (("THANK_YOU", "FINISHED"), "Thank you, I am finished."),
    (("NO", "STOP"), "No, stop."),
    (("YES", "PLEASE"), "Yes, please."),
    (("WANT", "MORE"), "I want more."),
)


@dataclass(frozen=True)
class VideoFeatureSummary:
    """Small set of features extracted from an uploaded or webcam clip."""

    input_name: str | None
    exists: bool
    file_size_bytes: int
    analyzer: str
    frame_count: int | None = None
    sampled_frames: int = 0
    fps: float | None = None
    duration_seconds: float | None = None
    average_brightness: float | None = None
    motion_score: float | None = None
    color_variation: float | None = None
    warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_space_poc(video: Any, mode: str, *, vlm_model_id: str) -> dict[str, Any]:
    """Run the hosted CNN/VLM comparison proof of concept."""

    started_at = time.perf_counter()
    selected_mode = _normalize_mode(mode)
    features = analyze_video(video)
    cnn_result = _run_cnn_poc(features) if selected_mode in {"cnn", "compare"} else {"status": "not_run"}
    vlm_result = (
        _run_vlm_poc(features, cnn_result, vlm_model_id)
        if selected_mode in {"vlm", "compare"}
        else {"status": "not_run"}
    )
    final_sentence = _select_final_sentence(cnn_result, vlm_result)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    return {
        "status": "proof_of_concept",
        "mode": selected_mode,
        "final_sentence": final_sentence,
        "speech_text": final_sentence,
        "latency_ms": latency_ms,
        "video_features": features.as_dict(),
        "cnn": cnn_result,
        "vlm": vlm_result,
        "comparison": _compare_outputs(cnn_result, vlm_result),
        "limitations": [
            "CNN output uses a hosted deterministic video-feature baseline until a real CNN artifact is trained.",
            "VLM output uses a grounded local fallback until Qwen2.5-VL is connected on GPU hardware.",
            "The Space is complete as a demo flow, not yet as a scientifically valid model benchmark.",
        ],
    }


def analyze_video(video: Any) -> VideoFeatureSummary:
    """Extract lightweight features from a Gradio video value."""

    path = _coerce_video_path(video)
    if path is None:
        return VideoFeatureSummary(
            input_name=None,
            exists=False,
            file_size_bytes=0,
            analyzer="none",
            warning="No video was provided. Using demo fallback output.",
        )

    exists = path.exists()
    file_size = path.stat().st_size if exists else 0
    if not exists:
        return VideoFeatureSummary(
            input_name=path.name,
            exists=False,
            file_size_bytes=0,
            analyzer="path",
            warning="Video path does not exist in the Space runtime.",
        )

    cv2_summary = _analyze_with_cv2(path, file_size)
    if cv2_summary:
        return cv2_summary

    return VideoFeatureSummary(
        input_name=path.name,
        exists=True,
        file_size_bytes=file_size,
        analyzer="file_metadata",
        average_brightness=_bounded_round((file_size % 997) / 997),
        motion_score=_bounded_round((file_size % 541) / 541),
        color_variation=_bounded_round((file_size % 389) / 389),
        warning="OpenCV could not read frames, so file metadata was used.",
    )


def result_to_markdown(result: dict[str, Any]) -> str:
    """Render a compact human-readable result for Gradio."""

    cnn = result.get("cnn", {})
    vlm = result.get("vlm", {})
    features = result.get("video_features", {})
    lines = [
        f"## Final Sentence",
        f"{result.get('final_sentence', 'No sentence available.')}",
        "",
        "## Model Outputs",
        f"- CNN: {cnn.get('prediction', cnn.get('status', 'not_run'))}",
        f"- VLM: {vlm.get('sentence', vlm.get('status', 'not_run'))}",
        "",
        "## Clip Features",
        f"- Analyzer: {features.get('analyzer')}",
        f"- Frames sampled: {features.get('sampled_frames')}",
        f"- Duration: {features.get('duration_seconds')} seconds",
        f"- Motion score: {features.get('motion_score')}",
        "",
        f"Latency: {result.get('latency_ms')} ms",
    ]
    return "\n".join(lines)


def _run_cnn_poc(features: VideoFeatureSummary) -> dict[str, Any]:
    gloss, sentence = _choose_sentence(features)
    confidence = _feature_confidence(features)
    return {
        "status": "poc_video_feature_baseline",
        "model_family": "sampled-frame-cnn-baseline",
        "prediction": " ".join(gloss),
        "gloss": list(gloss),
        "sentence_class": sentence,
        "confidence": confidence,
        "explanation": (
            "Hosted PoC CNN branch. It uses deterministic clip features now, "
            "and keeps the same output contract that a trained CNN model will use."
        ),
    }


def _run_vlm_poc(features: VideoFeatureSummary, cnn_result: dict[str, Any], model_id: str) -> dict[str, Any]:
    gloss = tuple(cnn_result.get("gloss") or _choose_sentence(features)[0])
    sentence = _gloss_to_sentence(gloss)
    confidence = max(0.35, round(float(cnn_result.get("confidence", 0.5)) - 0.04, 2))
    return {
        "status": "grounded_mock_vlm",
        "model_id": model_id,
        "gloss": list(gloss),
        "sentence": sentence,
        "confidence": confidence,
        "needs_clarification": confidence < 0.5,
        "explanation": (
            "Hosted PoC VLM branch. It simulates strict JSON sentence output "
            "grounded by the CNN/token trace until Qwen2.5-VL is attached."
        ),
    }


def _choose_sentence(features: VideoFeatureSummary) -> tuple[tuple[str, ...], str]:
    if not features.exists or features.sampled_frames < 2:
        return POC_SENTENCES[1]

    brightness = features.average_brightness or 0.5
    motion = features.motion_score or 0.0
    color = features.color_variation or 0.0

    if motion > 0.18 and brightness > 0.52:
        return POC_SENTENCES[0]
    if motion > 0.18:
        return POC_SENTENCES[5]
    if brightness < 0.34:
        return POC_SENTENCES[3]
    if color > 0.22:
        return POC_SENTENCES[2]
    return POC_SENTENCES[4]


def _gloss_to_sentence(gloss: tuple[str, ...]) -> str:
    for known_gloss, sentence in POC_SENTENCES:
        if tuple(gloss) == known_gloss:
            return sentence
    return " ".join(token.replace("_", " ").title() for token in gloss) + "."


def _feature_confidence(features: VideoFeatureSummary) -> float:
    if not features.exists:
        return 0.34
    frame_bonus = min((features.sampled_frames or 0) / 24, 1.0) * 0.22
    motion_bonus = min(features.motion_score or 0.0, 0.3) * 0.5
    size_bonus = min(math.log10(max(features.file_size_bytes, 1)) / 10, 0.15)
    return round(min(0.91, 0.42 + frame_bonus + motion_bonus + size_bonus), 2)


def _compare_outputs(cnn_result: dict[str, Any], vlm_result: dict[str, Any]) -> dict[str, Any]:
    if cnn_result.get("status") == "not_run" or vlm_result.get("status") == "not_run":
        return {"status": "partial", "agreement": None}
    cnn_sentence = str(cnn_result.get("sentence_class", "")).strip().lower()
    vlm_sentence = str(vlm_result.get("sentence", "")).strip().lower()
    return {
        "status": "compared",
        "agreement": cnn_sentence == vlm_sentence,
        "cnn_confidence": cnn_result.get("confidence"),
        "vlm_confidence": vlm_result.get("confidence"),
    }


def _select_final_sentence(cnn_result: dict[str, Any], vlm_result: dict[str, Any]) -> str:
    if vlm_result.get("sentence"):
        return str(vlm_result["sentence"])
    if cnn_result.get("sentence_class"):
        return str(cnn_result["sentence_class"])
    return POC_SENTENCES[1][1]


def _normalize_mode(mode: str) -> str:
    selected = str(mode or "Compare").strip().lower()
    return selected if selected in {"cnn", "vlm", "compare"} else "compare"


def _coerce_video_path(video: Any) -> Path | None:
    if video is None:
        return None
    if isinstance(video, dict):
        for key in ("video", "name", "path"):
            if video.get(key):
                return Path(str(video[key]))
        return None
    if isinstance(video, (tuple, list)):
        for item in video:
            path = _coerce_video_path(item)
            if path:
                return path
        return None
    text = str(video).strip()
    return Path(text) if text else None


def _analyze_with_cv2(path: Path, file_size: int) -> VideoFeatureSummary | None:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(reported_frame_count // 24, 1) if reported_frame_count else 1
    frame_index = 0
    sampled = 0
    brightness_values: list[float] = []
    color_values: list[float] = []
    motion_values: list[float] = []
    previous_gray = None

    while sampled < 24:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            resized = cv2.resize(frame, (48, 48))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(np.mean(gray) / 255.0))
            color_values.append(float(np.std(resized) / 255.0))
            if previous_gray is not None:
                diff = np.abs(gray.astype("float32") - previous_gray.astype("float32"))
                motion_values.append(float(np.mean(diff) / 255.0))
            previous_gray = gray
            sampled += 1
        frame_index += 1

    capture.release()
    if sampled == 0:
        return None

    duration = round(reported_frame_count / fps, 2) if fps > 0 and reported_frame_count else None
    return VideoFeatureSummary(
        input_name=path.name,
        exists=True,
        file_size_bytes=file_size,
        analyzer="opencv",
        frame_count=reported_frame_count or frame_index,
        sampled_frames=sampled,
        fps=round(fps, 2) if fps else None,
        duration_seconds=duration,
        average_brightness=_mean_or_none(brightness_values),
        motion_score=_mean_or_none(motion_values) or 0.0,
        color_variation=_mean_or_none(color_values),
    )


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return _bounded_round(sum(values) / len(values))


def _bounded_round(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 4)
