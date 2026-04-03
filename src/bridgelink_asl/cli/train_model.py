"""CLI for training the baseline centroid model."""

from __future__ import annotations

import argparse

from ..config import load_config
from ..training import train_centroid_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BridgeLink ASL centroid baseline.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--dataset", help="Path to a JSONL landmark dataset.")
    parser.add_argument("--output", help="Where to save the trained model.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(
        args.config,
        overrides={
            "dataset_path": args.dataset,
            "model_path": args.output,
        },
    )
    summary = train_centroid_model(config.dataset_path, config.model_path)
    print("BridgeLink ASL baseline training complete.")
    print(f"Dataset: {summary.dataset_path}")
    print(f"Saved model: {summary.output_path}")
    print(f"Training labels: {', '.join(summary.labels_trained)}")
    print(f"Training records used: {summary.records_used}")
    print(f"Split distribution: {summary.split_distribution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
