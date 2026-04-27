"""Generate report-ready metrics and SVG assets from a clip manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.project_assets import write_comparison_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BridgeLink ASL experiment/report artifacts.")
    parser.add_argument("--manifest", default="data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--vlm-results", help="Optional JSONL output from run_wrapper compare mode.")
    args = parser.parse_args()

    paths = write_comparison_artifacts(
        args.manifest,
        args.output_dir,
        vlm_results_path=args.vlm_results,
    )
    print("Generated report artifacts:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
