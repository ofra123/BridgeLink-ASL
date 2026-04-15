"""Evaluate the WLASL Transformer + VLM reranking experiment.

Example:
    python scripts/evaluate_hybrid_vlm.py \
        --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl \
        --output-dir results/vlm_eval

After filling the generated CSV's `vlm_prediction` column:
    python scripts/evaluate_hybrid_vlm.py \
        --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl \
        --predictions results/vlm_eval/vlm_review_template.csv \
        --output-dir results/vlm_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.hybrid_eval import (
    compute_hybrid_metrics,
    load_jsonl,
    merge_predictions,
    write_jsonl,
    write_review_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and score Transformer + VLM hybrid reranking metrics."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to wlasl25_hybrid_eval.jsonl from the Colab notebook.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/vlm_eval"),
        type=Path,
        help="Directory for metrics, merged JSONL, and review CSV outputs.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Optional CSV or JSONL containing video_id plus vlm_prediction. "
            "Use the generated review CSV after filling the vlm_prediction column."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.manifest)
    if args.predictions:
        rows = merge_predictions(rows, args.predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_csv = args.output_dir / "vlm_review_template.csv"
    merged_jsonl = args.output_dir / "vlm_hybrid_results.jsonl"
    metrics_json = args.output_dir / "vlm_hybrid_metrics.json"

    write_review_csv(rows, review_csv)
    write_jsonl(rows, merged_jsonl)
    metrics = compute_hybrid_metrics(rows)
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Samples: {metrics['num_samples']}")
    print(f"Classes: {metrics['num_classes']}")
    print(f"Candidate model: {metrics['candidate_model']}")
    print(f"Candidate top-1 accuracy: {metrics['candidate_top1_accuracy']:.3f}")
    print(f"Candidate top-5 coverage: {metrics['candidate_top5_coverage']:.3f}")
    if metrics["vlm_rerank_accuracy"] is None:
        print("VLM rerank accuracy: not scored yet")
        print(f"Fill the vlm_prediction column in: {review_csv}")
    else:
        print(f"VLM rerank accuracy: {metrics['vlm_rerank_accuracy']:.3f}")
    print(f"Wrote metrics: {metrics_json}")
    print(f"Wrote merged rows: {merged_jsonl}")


if __name__ == "__main__":
    main()
