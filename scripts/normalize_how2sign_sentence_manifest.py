"""Normalize duplicate / near-duplicate sentence labels in a How2Sign manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.sentence_labels import normalize_manifest_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize duplicate How2Sign sentence labels in a manifest.")
    parser.add_argument("--input", required=True, help="Input JSONL manifest path.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument(
        "--summary-output",
        help="Optional JSON summary path describing class merges and count changes.",
    )
    parser.add_argument(
        "--strategy",
        default="conservative",
        choices=("conservative",),
        help="Sentence normalization strategy to apply.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized_rows, summary = normalize_manifest_rows(rows, strategy=args.strategy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in normalized_rows) + "\n",
        encoding="utf-8",
    )

    payload = summary.as_dict()
    payload["input_manifest"] = str(input_path)
    payload["output_manifest"] = str(output_path)
    payload["strategy"] = args.strategy

    if args.summary_output:
        summary_path = Path(args.summary_output).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
