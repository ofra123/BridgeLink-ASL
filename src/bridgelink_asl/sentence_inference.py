"""Inference helpers for the How2Sign sentence-level 3D CNN."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SentenceClipRuntime:
    """Runtime wrapper for a Keras sentence clip classifier."""

    model: Any
    labels: list[str]
    frame_count: int
    image_size: int
    channels: int
    model_path: str

    def predict_clip(self, clip_volume: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """Run inference on one clip volume with shape (T, H, W, C)."""

        batch = np.expand_dims(clip_volume.astype(np.float32), axis=0)
        probabilities = np.asarray(self.model.predict(batch, verbose=0))[0]
        top5_idx = probabilities.argsort()[::-1][:5]
        top5 = [(self.labels[int(i)], float(probabilities[int(i)])) for i in top5_idx]
        best_idx = int(probabilities.argmax())
        return self.labels[best_idx], float(probabilities[best_idx]), top5


def load_sentence_runtime(
    local_model_path: str | Path = "models/cnn-3d-sentence-top25.keras",
    *,
    local_labels_path: str | Path | None = None,
    hf_repo: str | None = None,
    hf_model_filename: str | None = None,
    hf_labels_filename: str | None = None,
) -> SentenceClipRuntime:
    """Load a trained How2Sign sentence classifier from disk or the HF Hub."""

    tf = _require_tensorflow()
    model_path = Path(local_model_path)
    model_filename = hf_model_filename or model_path.name
    labels_path = Path(local_labels_path) if local_labels_path else model_path.with_suffix(".labels.json")
    labels_filename = hf_labels_filename or labels_path.name

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download

            model_path = Path(hf_hub_download(repo_id=hf_repo, filename=model_filename))
            labels_path = Path(hf_hub_download(repo_id=hf_repo, filename=labels_filename))
        except Exception as exc:
            print(f"[bridgelink] HF Hub download failed for sentence model ({exc}); falling back to local files")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Sentence CNN weights not found at {model_path}. "
            f"Upload {model_filename} to the repo or set HF_SENTENCE_MODEL_REPO."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Sentence CNN labels not found at {labels_path}. "
            f"Upload {labels_filename} to the repo or set HF_SENTENCE_MODEL_REPO."
        )

    model = tf.keras.models.load_model(model_path)
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = list(payload["labels"])
    input_shape = tuple(model.input_shape)
    if len(input_shape) != 5:
        raise ValueError(f"Unexpected sentence CNN input shape: {input_shape}")

    _, frame_count, image_height, image_width, channels = input_shape
    if image_height != image_width:
        raise ValueError(f"Sentence CNN expects square frames, got {input_shape}")

    return SentenceClipRuntime(
        model=model,
        labels=labels,
        frame_count=int(frame_count),
        image_size=int(image_height),
        channels=int(channels),
        model_path=str(model_path),
    )


def extract_clip_volume(
    video_path: str | Path,
    *,
    frame_count: int,
    image_size: int,
) -> tuple[np.ndarray | None, dict[str, float | int | None]]:
    """Decode a video file and convert it into a fixed-size RGB clip volume."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None, {"source_frames": 0, "sampled_frames": 0, "fps": None, "duration_seconds": None}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        return None, {"source_frames": 0, "sampled_frames": 0, "fps": fps or None, "duration_seconds": None}

    indexes = _uniform_indexes(len(frames), frame_count)
    clip_frames: list[np.ndarray] = []
    for index in indexes:
        rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        clip_frames.append(resized.astype(np.float32))

    duration_seconds = None
    if fps > 0:
        duration_seconds = round((reported_frame_count or len(frames)) / fps, 2)

    metadata: dict[str, float | int | None] = {
        "source_frames": len(frames),
        "sampled_frames": len(clip_frames),
        "fps": round(fps, 2) if fps > 0 else None,
        "duration_seconds": duration_seconds,
    }
    return np.stack(clip_frames, axis=0), metadata


def _uniform_indexes(total_frames: int, frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    return np.rint(np.linspace(0, total_frames - 1, frame_count)).astype(int)


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for the How2Sign sentence classifier. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc
    return tf
