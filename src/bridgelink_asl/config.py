"""Runtime configuration for BridgeLink ASL."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from .vocabulary import DEFAULT_DEMO_SEQUENCE

_ENV_FIELDS = {
    "camera_index": "BRIDGELINK_CAMERA_INDEX",
    "model_path": "BRIDGELINK_MODEL_PATH",
    "confidence_threshold": "BRIDGELINK_CONFIDENCE_THRESHOLD",
    "hold_frames": "BRIDGELINK_HOLD_FRAMES",
    "tts_provider": "BRIDGELINK_TTS_PROVIDER",
    "transcript_path": "BRIDGELINK_TRANSCRIPT_PATH",
    "metrics_path": "BRIDGELINK_METRICS_PATH",
    "dataset_path": "BRIDGELINK_DATASET_PATH",
    "clip_manifest_path": "BRIDGELINK_CLIP_MANIFEST_PATH",
    "max_frames": "BRIDGELINK_MAX_FRAMES",
    "frame_interval_seconds": "BRIDGELINK_FRAME_INTERVAL_SECONDS",
    "use_camera": "BRIDGELINK_USE_CAMERA",
    "demo_sequence": "BRIDGELINK_DEMO_SEQUENCE",
    "model_mode": "BRIDGELINK_MODEL_MODE",
    "cnn_model_path": "BRIDGELINK_CNN_MODEL_PATH",
    "cnn_frame_count": "BRIDGELINK_CNN_FRAME_COUNT",
    "cnn_image_size": "BRIDGELINK_CNN_IMAGE_SIZE",
    "cnn_batch_size": "BRIDGELINK_CNN_BATCH_SIZE",
    "cnn_epochs": "BRIDGELINK_CNN_EPOCHS",
    "vlm_provider": "BRIDGELINK_VLM_PROVIDER",
    "vlm_model_id": "BRIDGELINK_VLM_MODEL_ID",
    "elevenlabs_api_key": "ELEVENLABS_API_KEY",
    "elevenlabs_voice_id": "ELEVENLABS_VOICE_ID",
}


@dataclass(frozen=True)
class AppConfig:
    """Config shared across demo, training, and evaluation entrypoints."""

    camera_index: int = 0
    model_path: Path = field(default_factory=lambda: Path("models/dev-baseline.json"))
    confidence_threshold: float = 0.7
    hold_frames: int = 3
    tts_provider: str = "mock"
    transcript_path: Path = field(default_factory=lambda: Path("outputs/demo-transcript.jsonl"))
    metrics_path: Path = field(default_factory=lambda: Path("outputs/eval-metrics.json"))
    dataset_path: Path = field(default_factory=lambda: Path("data/processed/sample_landmarks.jsonl"))
    clip_manifest_path: Path = field(default_factory=lambda: Path("data/processed/sample_sentence_clips.jsonl"))
    max_frames: int = 18
    frame_interval_seconds: float = 0.0
    use_camera: bool = True
    demo_sequence: tuple[str, ...] = field(default_factory=lambda: DEFAULT_DEMO_SEQUENCE)
    model_mode: str = "word"
    cnn_model_path: Path = field(default_factory=lambda: Path("models/cnn-baseline.keras"))
    cnn_frame_count: int = 16
    cnn_image_size: int = 160
    cnn_batch_size: int = 4
    cnn_epochs: int = 8
    vlm_provider: str = "mock"
    vlm_model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize config values for logs or JSON output."""

        return {
            "camera_index": self.camera_index,
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "hold_frames": self.hold_frames,
            "tts_provider": self.tts_provider,
            "transcript_path": str(self.transcript_path),
            "metrics_path": str(self.metrics_path),
            "dataset_path": str(self.dataset_path),
            "clip_manifest_path": str(self.clip_manifest_path),
            "max_frames": self.max_frames,
            "frame_interval_seconds": self.frame_interval_seconds,
            "use_camera": self.use_camera,
            "demo_sequence": list(self.demo_sequence),
            "model_mode": self.model_mode,
            "cnn_model_path": str(self.cnn_model_path),
            "cnn_frame_count": self.cnn_frame_count,
            "cnn_image_size": self.cnn_image_size,
            "cnn_batch_size": self.cnn_batch_size,
            "cnn_epochs": self.cnn_epochs,
            "vlm_provider": self.vlm_provider,
            "vlm_model_id": self.vlm_model_id,
            "elevenlabs_api_key": "***" if self.elevenlabs_api_key else None,
            "elevenlabs_voice_id": self.elevenlabs_voice_id,
        }


def load_config(config_path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load config values from defaults, JSON, env vars, then explicit overrides."""

    defaults = AppConfig()
    raw_values = {item.name: getattr(defaults, item.name) for item in fields(AppConfig)}
    base_dir = Path.cwd()

    if config_path:
        config_file = Path(config_path).expanduser().resolve()
        base_dir = config_file.parent
        raw_values.update(json.loads(config_file.read_text(encoding="utf-8")))

    raw_values.update(_load_env_values())
    if overrides:
        raw_values.update({key: value for key, value in overrides.items() if value is not None})

    return AppConfig(
        camera_index=int(raw_values["camera_index"]),
        model_path=_coerce_path(raw_values["model_path"], base_dir),
        confidence_threshold=float(raw_values["confidence_threshold"]),
        hold_frames=int(raw_values["hold_frames"]),
        tts_provider=str(raw_values["tts_provider"]).strip().lower(),
        transcript_path=_coerce_path(raw_values["transcript_path"], base_dir),
        metrics_path=_coerce_path(raw_values["metrics_path"], base_dir),
        dataset_path=_coerce_path(raw_values["dataset_path"], base_dir),
        clip_manifest_path=_coerce_path(raw_values["clip_manifest_path"], base_dir),
        max_frames=int(raw_values["max_frames"]),
        frame_interval_seconds=float(raw_values["frame_interval_seconds"]),
        use_camera=_coerce_bool(raw_values["use_camera"]),
        demo_sequence=_coerce_sequence(raw_values["demo_sequence"]),
        model_mode=str(raw_values["model_mode"]).strip().lower(),
        cnn_model_path=_coerce_path(raw_values["cnn_model_path"], base_dir),
        cnn_frame_count=int(raw_values["cnn_frame_count"]),
        cnn_image_size=int(raw_values["cnn_image_size"]),
        cnn_batch_size=int(raw_values["cnn_batch_size"]),
        cnn_epochs=int(raw_values["cnn_epochs"]),
        vlm_provider=str(raw_values["vlm_provider"]).strip().lower(),
        vlm_model_id=str(raw_values["vlm_model_id"]).strip(),
        elevenlabs_api_key=_coerce_optional_string(raw_values.get("elevenlabs_api_key")),
        elevenlabs_voice_id=_coerce_optional_string(raw_values.get("elevenlabs_voice_id")),
    )


def _load_env_values() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field_name, env_name in _ENV_FIELDS.items():
        if env_name in os.environ:
            values[field_name] = os.environ[env_name]
    return values


def _coerce_path(value: str | Path, base_dir: Path) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _coerce_sequence(value: Any) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else value
    return tuple(str(item).strip().upper() for item in items if str(item).strip())


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
