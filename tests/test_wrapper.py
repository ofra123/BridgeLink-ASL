from __future__ import annotations

import json

from bridgelink_asl.asl_types import SentenceEvent
from bridgelink_asl.wrapper import run_wrapper


def test_compare_mode_logs_cnn_and_vlm_predictions(tmp_path) -> None:
    manifest_path = tmp_path / "clips.jsonl"
    output_path = tmp_path / "comparison-results.jsonl"
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

    assert summary.records_processed == 1
    assert summary.failures == 0
    assert rows[0]["clip_id"] == "team_hello_want_drink_001"
    assert rows[0]["cnn_prediction"]["model_mode"] == "cnn"
    assert rows[0]["vlm_prediction"]["model_mode"] == "vlm"
    assert "want" in rows[0]["cnn_prediction"]["sentence"].lower()
    assert rows[0]["vlm_prediction"]["sentence"] == "Hello, I want a drink."
    assert isinstance(rows[0]["cnn_latency_ms"], float)
    assert isinstance(rows[0]["vlm_latency_ms"], float)


def test_vlm_mode_falls_back_to_gloss_when_confidence_is_low(tmp_path) -> None:
    manifest_path = tmp_path / "clips.jsonl"
    output_path = tmp_path / "comparison-results.jsonl"
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

    assert summary.failures == 1
    assert row["cnn_prediction"] is None
    assert row["vlm_prediction"]["sentence"] == "No stop."
    assert row["vlm_prediction"]["needs_clarification"] is True
    assert any("gloss fallback" in note for note in row["failure_notes"])


def test_cnn_mode_only_writes_cnn_fields(tmp_path) -> None:
    manifest_path = tmp_path / "clips.jsonl"
    output_path = tmp_path / "comparison-results.jsonl"
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

    assert row["cnn_prediction"] is not None
    assert row["vlm_prediction"] is None
    assert isinstance(row["token_trace"], list)
    assert row["failure_notes"] == []
