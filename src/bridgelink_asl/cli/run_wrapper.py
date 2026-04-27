"""CLI for the phase-3 sentence wrapper modes."""

from __future__ import annotations

import argparse

from ..config import load_config
from ..wrapper import LocalQwen25VlmInterpreter, MockSentenceInterpreter, run_wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BridgeLink ASL sentence wrapper.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--manifest", help="Clip manifest JSONL path.")
    parser.add_argument(
        "--mode",
        choices=("cnn", "vlm", "compare"),
        default="compare",
        help="Wrapper mode to run over the clip manifest.",
    )
    parser.add_argument(
        "--output",
        default="outputs/comparison-results.jsonl",
        help="Where to write JSONL wrapper results.",
    )
    parser.add_argument(
        "--vlm-provider",
        choices=("mock", "local"),
        help="Override the configured VLM provider for vlm/compare modes.",
    )
    parser.add_argument(
        "--vlm-confidence-floor",
        type=float,
        default=0.6,
        help="Fallback to the gloss sentence below this confidence threshold.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overrides = {
        "clip_manifest_path": args.manifest,
        "vlm_provider": args.vlm_provider,
    }
    config = load_config(args.config, overrides=overrides)
    interpreter = _build_interpreter(config.vlm_provider, config.vlm_model_id)
    summary = run_wrapper(
        config.clip_manifest_path,
        mode=args.mode,
        output_path=args.output,
        interpreter=interpreter,
        vlm_confidence_floor=args.vlm_confidence_floor,
    )

    print("BridgeLink ASL wrapper complete.")
    print(f"Mode: {summary.mode}")
    print(f"Manifest: {summary.manifest_path}")
    print(f"Records processed: {summary.records_processed}")
    print(f"Failures: {summary.failures}")
    print(f"Output path: {summary.output_path}")
    return 0


def _build_interpreter(provider: str, model_id: str):
    if provider == "local":
        return LocalQwen25VlmInterpreter(model_id=model_id)
    return MockSentenceInterpreter()


if __name__ == "__main__":
    raise SystemExit(main())
