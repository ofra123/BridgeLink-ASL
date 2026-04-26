"""CLI for the optional sampled-frame CNN baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..clip_dataset import load_clip_dataset, validate_clip_dataset
from ..cnn import CnnModelConfig, build_cnn_training_plan, describe_cnn_baseline, train_clip_cnn_model
from ..config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or dry-run the BridgeLink ASL clip CNN baseline.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--clips", help="Path to a sentence clip JSONL manifest.")
    parser.add_argument("--output", help="Where to save the trained CNN model.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--frame-count", type=int, help="Number of sampled frames per clip.")
    parser.add_argument("--image-size", type=int, help="Square image size for sampled frames.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the CNN plan without TensorFlow.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app_config = load_config(
        args.config,
        overrides={
            "clip_manifest_path": args.clips,
            "cnn_model_path": args.output,
        },
    )
    cnn_config = CnnModelConfig(
        frame_count=args.frame_count or app_config.cnn_frame_count,
        image_size=args.image_size or app_config.cnn_image_size,
        batch_size=args.batch_size or app_config.cnn_batch_size,
        epochs=args.epochs or app_config.cnn_epochs,
        model_path=Path(app_config.cnn_model_path),
        manifest_path=Path(app_config.clip_manifest_path),
    )

    records = load_clip_dataset(cnn_config.manifest_path)
    issues = validate_clip_dataset(
        records,
        require_sampled_frames=not args.dry_run,
        require_files=not args.dry_run,
    )
    if issues:
        print("BridgeLink ASL CNN manifest validation failed.")
        for issue in issues:
            print(f"- {issue}")
        return 2

    plan = build_cnn_training_plan(records, cnn_config)
    if args.dry_run:
        payload = {
            "cnn": describe_cnn_baseline(cnn_config, num_classes=len(plan.labels)),
            "plan": plan.as_dict(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    summary = train_clip_cnn_model(cnn_config.manifest_path, cnn_config.model_path, cnn_config)
    print("BridgeLink ASL CNN training complete.")
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
