"""Clip-level dataset metadata for CNN and VLM comparison."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ClipDatasetRecord:
    """Metadata for one sentence-level ASL clip."""

    clip_id: str
    split: str
    source: str
    gloss: tuple[str, ...]
    english: str
    sentence_label: str | None = None
    video_path: Path | None = None
    sampled_frames: tuple[Path, ...] = ()
    landmarks_path: Path | None = None
    notes: str = ""

    @property
    def label(self) -> str:
        """Return the class label used by CNN metrics."""

        if self.sentence_label:
            return self.sentence_label
        return " ".join(self.gloss)


def load_clip_dataset(manifest_path: str | Path) -> list[ClipDatasetRecord]:
    """Load a JSONL clip manifest from disk."""

    path = Path(manifest_path).expanduser().resolve()
    records: list[ClipDatasetRecord] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        records.append(_record_from_payload(payload))
    return records


def validate_clip_dataset(
    records: Iterable[ClipDatasetRecord],
    *,
    require_sampled_frames: bool = False,
    require_files: bool = False,
) -> list[str]:
    """Return human-readable validation issues for a clip manifest."""

    materialized = list(records)
    issues: list[str] = []
    if not materialized:
        return ["Clip dataset is empty."]

    clip_ids = [record.clip_id for record in materialized]
    duplicate_ids = sorted(clip_id for clip_id, count in Counter(clip_ids).items() if count > 1)
    if duplicate_ids:
        issues.append(f"Clip dataset contains duplicate clip IDs: {', '.join(duplicate_ids)}.")

    bad_splits = sorted({record.split for record in materialized if record.split not in SUPPORTED_SPLITS})
    if bad_splits:
        issues.append(f"Clip dataset contains unsupported splits: {', '.join(bad_splits)}.")

    missing_gloss = sorted(record.clip_id for record in materialized if not record.gloss)
    if missing_gloss:
        issues.append(f"Clip dataset contains records without gloss labels: {', '.join(missing_gloss)}.")

    missing_english = sorted(record.clip_id for record in materialized if not record.english.strip())
    if missing_english:
        issues.append(f"Clip dataset contains records without English targets: {', '.join(missing_english)}.")

    non_uppercase_gloss = sorted(
        record.clip_id
        for record in materialized
        if any(token != token.upper() for token in record.gloss)
    )
    if non_uppercase_gloss:
        issues.append(f"Clip dataset contains non-uppercase gloss tokens: {', '.join(non_uppercase_gloss)}.")

    if require_sampled_frames:
        missing_frames = sorted(record.clip_id for record in materialized if not record.sampled_frames)
        if missing_frames:
            issues.append(f"Clip dataset contains records without sampled frame paths: {', '.join(missing_frames)}.")

    if require_files:
        missing_files = sorted(
            str(path)
            for record in materialized
            for path in (*record.sampled_frames, *((record.landmarks_path,) if record.landmarks_path else ()))
            if not path.exists()
        )
        if missing_files:
            issues.append(f"Clip dataset references missing files: {', '.join(missing_files)}.")

    return issues


def summarize_clip_splits(records: Iterable[ClipDatasetRecord]) -> dict[str, int]:
    """Count clip records by split."""

    return dict(Counter(record.split for record in records))


def _record_from_payload(payload: dict[str, Any]) -> ClipDatasetRecord:
    gloss = tuple(str(token).strip().upper() for token in payload.get("gloss", []) if str(token).strip())
    sampled_frames = tuple(
        path
        for raw_path in payload.get("sampled_frames", [])
        if (path := _optional_path(raw_path)) is not None
    )
    return ClipDatasetRecord(
        clip_id=str(payload["clip_id"]).strip(),
        split=str(payload["split"]).strip().lower(),
        source=str(payload.get("source", "unknown")).strip(),
        gloss=gloss,
        english=str(payload.get("english", "")).strip(),
        sentence_label=_optional_string(payload.get("sentence_label")),
        video_path=_optional_path(payload.get("video_path")),
        sampled_frames=sampled_frames,
        landmarks_path=_optional_path(payload.get("landmarks_path")),
        notes=str(payload.get("notes", "")).strip(),
    )


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
