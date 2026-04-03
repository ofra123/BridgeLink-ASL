"""CLI for evaluating a saved BridgeLink ASL model."""

from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..evaluation import evaluate_saved_model, save_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the BridgeLink ASL baseline model.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--dataset", help="Path to a JSONL landmark dataset.")
    parser.add_argument("--model", help="Path to a saved model.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--metrics-path", help="Where to save evaluation metrics.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(
        args.config,
        overrides={
            "dataset_path": args.dataset,
            "model_path": args.model,
            "metrics_path": args.metrics_path,
        },
    )
    metrics = evaluate_saved_model(config.model_path, config.dataset_path, split=args.split)
    saved_path = save_metrics(metrics, config.metrics_path)
    print("BridgeLink ASL evaluation complete.")
    print(json.dumps(metrics.as_dict(), indent=2))
    print(f"Metrics saved to: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
