from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .io_utils import read_jsonl, resolve_path
from .qwen_video_utils import build_messages


class VideoJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        video_column: str,
        target_column: str,
        prompt: str,
        video_fps: int | float | None = None,
        max_frames: int | None = None,
        max_samples: int | None = None,
        require_target: bool = True,
    ) -> None:
        rows = read_jsonl(jsonl_path)
        if max_samples is not None and max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows
        self.video_column = video_column
        self.target_column = target_column
        self.prompt = prompt
        self.video_fps = video_fps
        self.max_frames = max_frames
        self.require_target = require_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        if self.video_column not in row:
            raise KeyError(f"Missing video column '{self.video_column}' in row id={row.get('id', idx)}")
        video_path = resolve_path(row[self.video_column])
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for row id={row.get('id', idx)}: {video_path}")
        target = row.get(self.target_column)
        if self.require_target and (target is None or str(target).strip() == ""):
            raise ValueError(f"Missing target '{self.target_column}' in row id={row.get('id', idx)}")
        return {
            "id": row.get("id", str(idx)),
            "video_path": str(video_path),
            "target": str(target) if target is not None else None,
            "messages": build_messages(
                video_path,
                self.prompt,
                target,
                video_fps=self.video_fps,
                max_frames=self.max_frames,
            ),
        }
