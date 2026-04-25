from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from bridgelink_asl.asl_types import SentenceEvent
from bridgelink_asl.wrapper import run_wrapper


class WrapperTests(unittest.TestCase):
    def test_compare_mode_logs_cnn_and_vlm_predictions(self) -> None:
        scratch_dir = _make_scratch_dir("wrapper-compare")
        manifest_path = scratch_dir / "clips.jsonl"
        output_path = scratch_dir / "comparison-results.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "clip_id": "team_hello_want_drink_001",
                    "split": "test",
                    "source": "team",
                    "gloss": ["HELLO", "WANT", "DRINK"],
                    "english": "Hello, I want a drink.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        summary = run_wrapper(manifest_path, mode="compare", output_path=output_path)
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(summary.records_processed, 1)
        self.assertEqual(summary.failures, 0)
        self.assertEqual(rows[0]["clip_id"], "team_hello_want_drink_001")
        self.assertEqual(rows[0]["cnn_prediction"]["model_mode"], "cnn")
        self.assertEqual(rows[0]["vlm_prediction"]["model_mode"], "vlm")
        self.assertIn("want", rows[0]["cnn_prediction"]["sentence"].lower())
        self.assertEqual(rows[0]["vlm_prediction"]["sentence"], "Hello, I want a drink.")
        self.assertIsInstance(rows[0]["cnn_latency_ms"], float)
        self.assertIsInstance(rows[0]["vlm_latency_ms"], float)

    def test_vlm_mode_falls_back_to_gloss_when_confidence_is_low(self) -> None:
        scratch_dir = _make_scratch_dir("wrapper-vlm-fallback")
        manifest_path = scratch_dir / "clips.jsonl"
        output_path = scratch_dir / "comparison-results.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "clip_id": "team_no_stop_001",
                    "split": "test",
                    "source": "team",
                    "gloss": ["NO", "STOP"],
                    "english": "No, stop.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class LowConfidenceInterpreter:
            def interpret(self, window):
                return SentenceEvent(
                    gloss=tuple(token.label for token in window.token_trace),
                    sentence="I am not sure.",
                    confidence=0.2,
                    model_mode="vlm",
                )

        summary = run_wrapper(
            manifest_path,
            mode="vlm",
            output_path=output_path,
            interpreter=LowConfidenceInterpreter(),
            vlm_confidence_floor=0.6,
        )
        row = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(summary.failures, 1)
        self.assertIsNone(row["cnn_prediction"])
        self.assertEqual(row["vlm_prediction"]["sentence"], "No stop.")
        self.assertTrue(row["vlm_prediction"]["needs_clarification"])
        self.assertTrue(any("gloss fallback" in note for note in row["failure_notes"]))

    def test_cnn_mode_only_writes_cnn_fields(self) -> None:
        scratch_dir = _make_scratch_dir("wrapper-cnn")
        manifest_path = scratch_dir / "clips.jsonl"
        output_path = scratch_dir / "comparison-results.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "clip_id": "team_please_help_001",
                    "split": "test",
                    "source": "team",
                    "gloss": ["PLEASE", "HELP"],
                    "english": "Please help.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        run_wrapper(manifest_path, mode="cnn", output_path=output_path)
        row = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertIsNotNone(row["cnn_prediction"])
        self.assertIsNone(row["vlm_prediction"])
        self.assertIsInstance(row["token_trace"], list)
        self.assertEqual(row["failure_notes"], [])


def _make_scratch_dir(prefix: str) -> Path:
    tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    scratch_dir = tmp_root / f"{prefix}-{uuid.uuid4().hex}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir
