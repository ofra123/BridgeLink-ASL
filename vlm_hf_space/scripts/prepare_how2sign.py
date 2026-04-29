from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VIDEO_COLUMNS = [
    "sentence_name", "SENTENCE_NAME",
    "video_path", "path", "file", "filename", "video",
    "video_name", "VIDEO_NAME",
    "sentence_id", "SENTENCE_ID",
    "clip_id", "CLIP_ID",
]

TARGET_COLUMNS = [
    "translation", "text", "sentence", "transcript",
    "raw_text", "normalized_text",
    "TRANSLATION", "TEXT", "SENTENCE", "TRANSCRIPT",
]

ID_COLUMNS = [
    "id", "sentence_id", "clip_id", "video_id",
    "ID", "SENTENCE_ID", "CLIP_ID", "VIDEO_ID", "SENTENCE_NAME",
]


def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    raise ValueError(
        f"Could not find {label} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def build_video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}

    for path in video_root.rglob("*.mp4"):
        keys = {
            path.name,
            path.stem,
            path.stem.replace("-rgb_front", ""),
            path.name.replace("-rgb_front.mp4", ""),
        }

        for key in keys:
            index[key] = path

    return index


def find_video(value: object, video_root: Path, index: dict[str, Path]) -> Path:
    raw = str(value).strip()
    name = Path(raw).name
    stem = Path(raw).stem

    candidates = [
        raw,
        name,
        stem,
        f"{raw}.mp4",
        f"{name}.mp4",
        f"{raw}-rgb_front.mp4",
        f"{name}-rgb_front.mp4",
        raw.replace("-rgb_front", ""),
        stem.replace("-rgb_front", ""),
    ]

    for candidate in candidates:
        if candidate in index:
            return index[candidate]

    direct_path = video_root / name
    if direct_path.exists():
        return direct_path

    raise FileNotFoundError(f"Could not find video for '{raw}' under {video_root}")


def write_split(csv_path: Path, video_root: Path, out_path: Path, split: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    if not video_root.exists():
        raise FileNotFoundError(f"Missing video directory: {video_root}")

    df = pd.read_csv(csv_path, sep="\t")
    print(f"{split} columns: {list(df.columns)}")

    video_col = pick_column(df, VIDEO_COLUMNS, "video")
    target_col = pick_column(df, TARGET_COLUMNS, "translation/text")

    id_col = None
    try:
        id_col = pick_column(df, ID_COLUMNS, "id")
    except ValueError:
        pass

    print(f"{split}: using video column = {video_col}")
    print(f"{split}: using target column = {target_col}")
    print(f"{split}: using id column = {id_col}")

    video_index = build_video_index(video_root)
    print(f"{split}: indexed {len(video_index)} video lookup keys")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_missing_video = 0
    skipped_empty_target = 0
    missing_examples = []

    missing_report_path = out_path.with_suffix(".missing_videos.txt")

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            target = str(row[target_col]).strip()

            if not target or target.lower() == "nan":
                skipped_empty_target += 1
                continue

            try:
                video_path = find_video(row[video_col], video_root, video_index)
            except FileNotFoundError:
                skipped_missing_video += 1
                if len(missing_examples) < 50:
                    missing_examples.append(str(row[video_col]))
                continue

            example = {
                "id": str(row[id_col]).strip() if id_col else str(i),
                "video_path": str(video_path),
                "translation": target,
                "split": split,
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    with missing_report_path.open("w", encoding="utf-8") as f:
        for item in missing_examples:
            f.write(item + "\n")

    total = len(df)
    print(f"{split}: total metadata rows = {total}")
    print(f"{split}: wrote usable rows = {written}")
    print(f"{split}: skipped missing videos = {skipped_missing_video}")
    print(f"{split}: skipped empty targets = {skipped_empty_target}")
    print(f"{split}: missing-video examples saved to {missing_report_path}")

    if total > 0:
        kept_pct = 100.0 * written / total
        skipped_pct = 100.0 * skipped_missing_video / total
        print(f"{split}: kept {kept_pct:.2f}% of rows")
        print(f"{split}: missing-video skip rate {skipped_pct:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_metadata", required=True)
    parser.add_argument("--val_metadata", required=True)
    parser.add_argument("--train_video_root", required=True)
    parser.add_argument("--val_video_root", required=True)
    parser.add_argument("--out_train", default="data/how2sign_train.jsonl")
    parser.add_argument("--out_val", default="data/how2sign_val.jsonl")

    args = parser.parse_args()

    write_split(
        csv_path=Path(args.train_metadata),
        video_root=Path(args.train_video_root),
        out_path=Path(args.out_train),
        split="train",
    )

    write_split(
        csv_path=Path(args.val_metadata),
        video_root=Path(args.val_video_root),
        out_path=Path(args.out_val),
        split="val",
    )


if __name__ == "__main__":
    main()