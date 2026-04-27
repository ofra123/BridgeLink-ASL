"""Build a closed-vocabulary How2Sign sentence manifest for 3D CNN training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.how2sign import write_how2sign_sentence_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a filtered How2Sign sentence manifest.")
    parser.add_argument(
        "--translation-dir",
        default="data/raw/how2sign/translations",
        help="Directory containing how2sign_realigned_{train,val,test}.csv files.",
    )
    parser.add_argument(
        "--clip-dir",
        default="data/raw/how2sign/clips/raw_videos",
        help="Directory containing pre-cut frontal RGB clip mp4 files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/how2sign_sentences_top12.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument("--min-count", type=int, default=8, help="Minimum clip count required for a sentence class.")
    parser.add_argument("--max-classes", type=int, default=12, help="Maximum number of sentence classes to keep.")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=20,
        help="Cap the number of clips kept per sentence class.",
    )
    args = parser.parse_args()

    summary = write_how2sign_sentence_manifest(
        args.output,
        translation_dir=args.translation_dir,
        clip_dir=args.clip_dir,
        min_count=args.min_count,
        max_classes=args.max_classes,
        max_samples_per_class=args.max_samples_per_class,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
