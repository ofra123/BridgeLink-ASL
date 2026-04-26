"""Helpers for building a closed-vocabulary How2Sign sentence manifest."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class How2SignSentenceRecord:
    clip_id: str
    split: str
    source: str
    sentence_label: str
    gloss: tuple[str, ...]
    english: str
    video_path: Path
    notes: str = ""

    def as_manifest_row(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "split": self.split,
            "source": self.source,
            "sentence_label": self.sentence_label,
            "gloss": list(self.gloss),
            "english": self.english,
            "video_path": str(self.video_path).replace("\\", "/"),
            "sampled_frames": [],
            "landmarks_path": None,
            "notes": self.notes,
        }


def build_how2sign_sentence_manifest(
    *,
    translation_dir: str | Path,
    clip_dir: str | Path,
    min_count: int = 8,
    max_classes: int = 12,
    max_samples_per_class: int = 20,
) -> tuple[list[How2SignSentenceRecord], dict[str, Any]]:
    """Build a closed-vocabulary sentence manifest from repeated How2Sign clips."""

    translation_root = Path(translation_dir).expanduser().resolve()
    clip_root = Path(clip_dir).expanduser().resolve()
    rows = _load_translation_rows(translation_root, clip_root)

    matched_rows = [row for row in rows if row["video_path"] is not None]
    sentence_counts = Counter(_normalize_sentence(row["english"]) for row in matched_rows)
    candidate_sentences = [
        sentence
        for sentence, count in sorted(sentence_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ][:max_classes]

    selected: list[How2SignSentenceRecord] = []
    per_class_counts: Counter[str] = Counter()
    for row in matched_rows:
        sentence_key = _normalize_sentence(row["english"])
        if sentence_key not in candidate_sentences:
            continue
        if per_class_counts[sentence_key] >= max_samples_per_class:
            continue
        per_class_counts[sentence_key] += 1
        selected.append(
            How2SignSentenceRecord(
                clip_id=str(row["video_name"]),
                split=str(row["split"]),
                source="how2sign-realigned",
                sentence_label=row["english"],
                gloss=_sentence_to_gloss(row["english"]),
                english=row["english"],
                video_path=row["video_path"],
                notes=f"How2Sign sentence clip from {row['video_name']}.",
            )
        )

    original_split_counts = dict(Counter(record.split for record in selected))
    selected = _rebalance_sentence_splits(selected)
    summary = {
        "translation_rows": len(rows),
        "matched_clip_rows": len(matched_rows),
        "selected_rows": len(selected),
        "selected_classes": len({record.sentence_label for record in selected}),
        "min_count": min_count,
        "max_classes": max_classes,
        "max_samples_per_class": max_samples_per_class,
        "top_sentence_counts": [
            {"sentence": sentence, "count": sentence_counts[sentence]}
            for sentence in candidate_sentences
        ],
        "original_split_counts": original_split_counts,
        "split_counts": dict(Counter(record.split for record in selected)),
        "split_strategy": "deterministic_per_class_70_15_15",
    }
    return selected, summary


def write_how2sign_sentence_manifest(
    output_path: str | Path,
    *,
    translation_dir: str | Path,
    clip_dir: str | Path,
    min_count: int = 8,
    max_classes: int = 12,
    max_samples_per_class: int = 20,
) -> dict[str, Any]:
    """Build and write the filtered sentence manifest to JSONL."""

    rows, summary = build_how2sign_sentence_manifest(
        translation_dir=translation_dir,
        clip_dir=clip_dir,
        min_count=min_count,
        max_classes=max_classes,
        max_samples_per_class=max_samples_per_class,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row.as_manifest_row()) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary["output_path"] = str(output)
    return summary


def _load_translation_rows(translation_root: Path, clip_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        csv_path = translation_root / f"how2sign_realigned_{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                sentence = str(row.get("SENTENCE", "")).strip()
                sentence_name = str(row.get("SENTENCE_NAME", "")).strip()
                if not sentence or not sentence_name:
                    continue
                video_path = clip_root / f"{sentence_name}.mp4"
                rows.append(
                    {
                        "split": split,
                        "sentence_id": str(row.get("SENTENCE_ID", sentence_name)).strip(),
                        "video_name": sentence_name,
                        "english": sentence,
                        "video_path": video_path if video_path.exists() else None,
                    }
                )
    return rows


def _normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.strip().split())


def _sentence_to_gloss(sentence: str) -> tuple[str, ...]:
    tokens = [match.group(0).upper() for match in TOKEN_PATTERN.finditer(sentence)]
    return tuple(tokens[:10] or ["SENTENCE"])


def _rebalance_sentence_splits(rows: list[How2SignSentenceRecord]) -> list[How2SignSentenceRecord]:
    grouped: dict[str, list[How2SignSentenceRecord]] = {}
    for row in rows:
        grouped.setdefault(row.sentence_label, []).append(row)

    balanced: list[How2SignSentenceRecord] = []
    for sentence_label, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: (item.clip_id, item.video_path.name))
        total = len(ordered)

        if total >= 3:
            val_count = max(1, round(total * 0.15))
            test_count = max(1, round(total * 0.15))
            if val_count + test_count >= total:
                val_count = 1
                test_count = 1
            train_count = total - val_count - test_count
            if train_count <= 0:
                train_count = max(1, total - 2)
                remaining = total - train_count
                val_count = 1 if remaining >= 1 else 0
                test_count = remaining - val_count
        elif total == 2:
            train_count, val_count, test_count = 1, 1, 0
        else:
            train_count, val_count, test_count = 1, 0, 0

        for index, item in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            balanced.append(replace(item, split=split))

    return balanced
