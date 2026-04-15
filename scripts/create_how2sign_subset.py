"""Create a BridgeLink clip manifest from downloaded How2Sign metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TEXT_COLUMNS = ("english", "translation", "text", "sentence", "SENTENCE", "SENTENCE_TEXT")
ID_COLUMNS = ("clip_id", "sentence_id", "SENTENCE_ID", "SENTENCE_NAME", "video_id", "VIDEO_ID", "name")
VIDEO_COLUMNS = ("video_path", "video", "VIDEO", "VIDEO_NAME", "file", "filename", "path")
SPLIT_COLUMNS = ("split", "SPLIT", "partition")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a small How2Sign subset JSONL manifest.")
    parser.add_argument("--metadata", nargs="+", required=True, help="How2Sign CSV/TSV metadata files.")
    parser.add_argument("--video-root", default="data/raw/how2sign", help="Local directory containing downloaded videos.")
    parser.add_argument("--output", default="data/processed/how2sign_subset.jsonl", help="Output JSONL manifest.")
    parser.add_argument("--max-per-split", type=int, default=40, help="Maximum clips to keep per split.")
    parser.add_argument("--keyword", action="append", default=[], help="Optional English keyword filter. Repeatable.")
    parser.add_argument("--label-column", help="Optional column to use as the CNN sentence label.")
    args = parser.parse_args()

    rows = []
    split_counts: dict[str, int] = {}
    keywords = [keyword.lower() for keyword in args.keyword]
    for metadata_file in args.metadata:
        split_hint = _split_from_name(Path(metadata_file).stem)
        for row in _read_table(Path(metadata_file)):
            split = _pick(row, SPLIT_COLUMNS) or split_hint
            split = _normalize_split(split)
            if split_counts.get(split, 0) >= args.max_per_split:
                continue

            english = _pick(row, TEXT_COLUMNS)
            if not english:
                continue
            if keywords and not any(keyword in english.lower() for keyword in keywords):
                continue

            clip_id = _pick(row, ID_COLUMNS) or f"how2sign_{split}_{split_counts.get(split, 0):04d}"
            video_name = _pick(row, VIDEO_COLUMNS) or f"{clip_id}.mp4"
            label = _pick(row, (args.label_column,)) if args.label_column else _label_from_text(english)
            payload = {
                "clip_id": _slug(clip_id),
                "split": split,
                "source": "how2sign-subset",
                "sentence_label": label,
                "gloss": _gloss_from_text(english),
                "english": english.strip(),
                "video_path": str(Path(args.video_root) / Path(video_name).name),
                "sampled_frames": [],
                "landmarks_path": None,
                "notes": "Generated from downloaded How2Sign metadata. Keep raw videos outside Git.",
            }
            rows.append(payload)
            split_counts[split] = split_counts.get(split, 0) + 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} How2Sign subset records to {output_path}")
    print(f"Split counts: {split_counts}")
    return 0


def _read_table(path: Path) -> list[dict[str, str]]:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _pick(row: dict[str, Any], columns: tuple[str | None, ...]) -> str | None:
    for column in columns:
        if not column:
            continue
        if column in row and str(row[column]).strip():
            return str(row[column]).strip()
    lower_lookup = {key.lower(): value for key, value in row.items()}
    for column in columns:
        if column and column.lower() in lower_lookup and str(lower_lookup[column.lower()]).strip():
            return str(lower_lookup[column.lower()]).strip()
    return None


def _split_from_name(name: str) -> str:
    lowered = name.lower()
    if "val" in lowered or "dev" in lowered:
        return "val"
    if "test" in lowered:
        return "test"
    return "train"


def _normalize_split(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"validation", "valid", "dev"}:
        return "val"
    if lowered in {"testing"}:
        return "test"
    return "train" if lowered not in {"train", "val", "test"} else lowered


def _label_from_text(text: str) -> str:
    tokens = _gloss_from_text(text)[:5]
    return "HOW2SIGN_" + "_".join(tokens or ["CLIP"])


def _gloss_from_text(text: str) -> list[str]:
    stopwords = {"THE", "A", "AN", "TO", "OF", "AND", "OR", "IN", "ON", "FOR", "YOU", "YOUR", "THIS", "THAT"}
    tokens = re.findall(r"[A-Za-z]+", text.upper())
    filtered = [token for token in tokens if token not in stopwords]
    return filtered[:8] or ["HOW2SIGN"]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return slug.strip("_") or "how2sign_clip"


if __name__ == "__main__":
    raise SystemExit(main())
