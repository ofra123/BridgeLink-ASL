"""Frame sources for live and synthetic demo runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from .config import AppConfig
from .types import Frame

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass
class SyntheticFrameSource:
    """Deterministic fallback source used in tests and no-camera mode."""

    frame_interval_seconds: float = 0.0
    name: str = "synthetic"

    def frames(self, max_frames: int) -> Iterator[Frame]:
        for index in range(max_frames):
            yield Frame(index=index, source=self.name, timestamp=time.time())
            if self.frame_interval_seconds:
                time.sleep(self.frame_interval_seconds)


@dataclass
class CameraFrameSource:
    """OpenCV-based webcam source when camera dependencies are available."""

    camera_index: int
    frame_interval_seconds: float = 0.0
    name: str = "camera"

    def is_available(self) -> bool:
        if cv2 is None:  # pragma: no cover - optional dependency
            return False
        capture = cv2.VideoCapture(self.camera_index)
        try:
            return bool(capture.isOpened())
        finally:
            capture.release()

    def frames(self, max_frames: int) -> Iterator[Frame]:
        if cv2 is None:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenCV is not installed.")

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open camera index {self.camera_index}.")

        try:
            for index in range(max_frames):
                ok, image = capture.read()
                if not ok:
                    break
                yield Frame(index=index, source=self.name, image=image, timestamp=time.time())
                if self.frame_interval_seconds:
                    time.sleep(self.frame_interval_seconds)
        finally:
            capture.release()


def build_frame_source(config: AppConfig) -> SyntheticFrameSource | CameraFrameSource:
    """Return the best available frame source for the current config."""

    camera_source = CameraFrameSource(
        camera_index=config.camera_index,
        frame_interval_seconds=config.frame_interval_seconds,
    )
    if config.use_camera and camera_source.is_available():
        return camera_source
    return SyntheticFrameSource(frame_interval_seconds=config.frame_interval_seconds)
