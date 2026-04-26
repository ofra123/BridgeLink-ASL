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
    candidate_labels: tuple[str, ...] = ()
    vlm_prompt: str | None = None
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
        records.append(_record_from_payload(payload, manifest_dir=path.parent))
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


def _record_from_payload(payload: dict[str, Any], *, manifest_dir: Path) -> ClipDatasetRecord:
    clip_id = _required_string(payload, "clip_id", fallback_key="video_id")
    split = str(payload.get("split", "test")).strip().lower() or "test"
    source = str(payload.get("source") or payload.get("candidate_model") or "unknown").strip()
    gloss = _coerce_gloss(payload)
    sampled_frames = tuple(
        path
        for raw_path in payload.get("sampled_frames", [])
        if (path := _resolve_media_path(raw_path, manifest_dir)) is not None
    )
    return ClipDatasetRecord(
        clip_id=clip_id,
        split=split,
        source=source,
        gloss=gloss,
        english=_coerce_english(payload),
        sentence_label=_optional_string(payload.get("sentence_label") or payload.get("true_label")),
        video_path=_resolve_media_path(payload.get("video_path"), manifest_dir, fallback_subdir="clips"),
        sampled_frames=sampled_frames,
        landmarks_path=_resolve_media_path(
            payload.get("landmarks_path") or payload.get("landmark_path"),
            manifest_dir,
        ),
        candidate_labels=_coerce_candidate_labels(payload),
        vlm_prompt=_optional_string(payload.get("vlm_prompt")),
        notes=str(payload.get("notes") or payload.get("vlm_prompt") or "").strip(),
    )


def _coerce_gloss(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_gloss = payload.get("gloss")
    if raw_gloss is None:
        candidate = payload.get("cnn_top1") or payload.get("model_top1") or payload.get("true_label")
        raw_gloss = [candidate] if candidate else []
    if isinstance(raw_gloss, str):
        raw_gloss = [raw_gloss]
    return tuple(str(token).strip().upper() for token in raw_gloss if str(token).strip())


def _coerce_english(payload: dict[str, Any]) -> str:
    english = str(payload.get("english", "")).strip()
    if english:
        return english
    return str(payload.get("true_label") or payload.get("sentence_label") or "").strip()


def _coerce_candidate_labels(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_candidates = payload.get("cnn_top5") or payload.get("model_top5") or ()
    labels: list[str] = []
    for item in raw_candidates:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
        else:
            label = str(item).strip()
        if label:
            labels.append(label.upper())
    if not labels:
        labels.extend(_coerce_gloss(payload))
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return tuple(ordered)


def _resolve_media_path(value: Any, manifest_dir: Path, fallback_subdir: str | None = None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (manifest_dir / candidate).resolve()
        if candidate.exists():
            return candidate
    elif candidate.exists():
        return candidate

    basename = Path(text).name
    search_roots = [manifest_dir]
    if fallback_subdir:
        search_roots.insert(0, manifest_dir / fallback_subdir)
    for root in search_roots:
        alternate = (root / basename).resolve()
        if alternate.exists():
            return alternate

    return candidate if candidate.is_absolute() else (manifest_dir / Path(text)).resolve()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_string(payload: dict[str, Any], key: str, *, fallback_key: str | None = None) -> str:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise KeyError(key)
    return text
