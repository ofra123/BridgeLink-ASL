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
    "max_frames": "BRIDGELINK_MAX_FRAMES",
    "frame_interval_seconds": "BRIDGELINK_FRAME_INTERVAL_SECONDS",
    "use_camera": "BRIDGELINK_USE_CAMERA",
    "demo_sequence": "BRIDGELINK_DEMO_SEQUENCE",
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
    max_frames: int = 18
    frame_interval_seconds: float = 0.0
    use_camera: bool = True
    demo_sequence: tuple[str, ...] = field(default_factory=lambda: DEFAULT_DEMO_SEQUENCE)
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
            "max_frames": self.max_frames,
            "frame_interval_seconds": self.frame_interval_seconds,
            "use_camera": self.use_camera,
            "demo_sequence": list(self.demo_sequence),
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
        max_frames=int(raw_values["max_frames"]),
        frame_interval_seconds=float(raw_values["frame_interval_seconds"]),
        use_camera=_coerce_bool(raw_values["use_camera"]),
        demo_sequence=_coerce_sequence(raw_values["demo_sequence"]),
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
