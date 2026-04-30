from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_config, resolve_path, write_jsonl
from src.label_utils import normalize_label


VIDEO_COLUMNS = ["video_path", "path", "file", "filename", "video"]
LABEL_COLUMNS = ["label", "gloss", "sign", "sign_label", "text"]
SPLIT_COLUMNS = ["split", "subset"]
ID_COLUMNS = ["id", "sign_id"]


def pick_column(columns: list[str], candidates: list[str], name: str) -> str:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"Missing {name} column. Tried: {', '.join(candidates)}")


def read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported metadata format: {path.suffix}. Use CSV, JSON, or JSONL.")


def normalize_split(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"val", "valid", "validation", "dev"}:
        return "val"
    if value in {"test", "testing"}:
        return "test"
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--out_train", default=None)
    parser.add_argument("--out_val", default=None)
    parser.add_argument("--skip_missing_check", action="store_true")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    out_train = args.out_train or cfg.get("train_jsonl", "data/asl_citizen_train.jsonl")
    out_val = args.out_val or cfg.get("val_jsonl", "data/asl_citizen_val.jsonl")

    metadata_path = resolve_path(args.metadata, Path.cwd())
    video_root = resolve_path(args.video_root, Path.cwd())
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    df = read_metadata(metadata_path)
    columns = list(df.columns)
    video_col = pick_column(columns, VIDEO_COLUMNS, "video path")
    label_col = pick_column(columns, LABEL_COLUMNS, "label")
    id_col = next((c for c in ID_COLUMNS if c in columns), None)
    split_col = next((c for c in SPLIT_COLUMNS if c in columns), None)

    rows = []
    for idx, row in df.iterrows():
        label = normalize_label(row[label_col])
        if not label:
            raise ValueError(f"Missing label in metadata row {idx}")
        raw_video = Path(str(row[video_col]))
        video_path = raw_video if raw_video.is_absolute() else video_root / raw_video
        if not args.skip_missing_check and not video_path.exists():
            raise FileNotFoundError(f"Missing video at metadata row {idx}: {video_path}")
        split = normalize_split(row[split_col]) if split_col else None
        rows.append(
            {
                "id": str(row[id_col]) if id_col else str(idx),
                "video_path": str(video_path),
                "label": label,
                "split": split,
            }
        )

    if split_col:
        train_rows = [r for r in rows if r["split"] == "train"]
        val_rows = [r for r in rows if r["split"] in {"val", "validation", "dev"}]
        if not train_rows or not val_rows:
            raise ValueError("Found a split column, but could not find both train and val rows.")
    else:
        split_at = max(1, int(len(rows) * (1 - args.val_fraction)))
        train_rows = rows[:split_at]
        val_rows = rows[split_at:]
        for r in train_rows:
            r["split"] = "train"
        for r in val_rows:
            r["split"] = "val"

    write_jsonl(out_train, train_rows)
    write_jsonl(out_val, val_rows)
    print(f"Wrote {len(train_rows)} train rows to {out_train}")
    print(f"Wrote {len(val_rows)} val rows to {out_val}")


if __name__ == "__main__":
    main()
