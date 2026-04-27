from __future__ import annotations

import json

from bridgelink_asl.clip_dataset import load_clip_dataset
from bridgelink_asl.asl_types import GestureWindow, SentenceEvent
from bridgelink_asl.wrapper import LocalQwen25VlmInterpreter, run_wrapper


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


def test_load_clip_dataset_accepts_hybrid_eval_rows_and_resolves_local_clip(tmp_path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    local_clip = clips_dir / "12320_computer.mp4"
    local_clip.write_bytes(b"fake")
    manifest_path = tmp_path / "hybrid.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "candidate_model": "landmark_cnn",
                "video_id": "12320",
                "true_label": "computer",
                "video_path": "/content/drive/MyDrive/BridgeLink-ASL/vlm_eval_wlasl25_cnn/clips/12320_computer.mp4",
                "cnn_top1": "computer",
                "cnn_top5": [
                    {"label": "computer", "confidence": 0.19},
                    {"label": "snow", "confidence": 0.08},
                ],
                "vlm_prompt": "Choose the best label from the list only.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_clip_dataset(manifest_path)

    assert len(records) == 1
    assert records[0].clip_id == "12320"
    assert records[0].split == "test"
    assert records[0].gloss == ("COMPUTER",)
    assert records[0].candidate_labels == ("COMPUTER", "SNOW")
    assert records[0].video_path == local_clip.resolve()


def test_local_qwen_interpreter_parses_json_response_without_real_model() -> None:
    class FakeInterpreter(LocalQwen25VlmInterpreter):
        def _generate_response_text(self, messages):
            return json.dumps(
                {
                    "gloss": ["computer"],
                    "sentence": "Computer.",
                    "confidence": 0.88,
                    "needs_clarification": False,
                }
            )

    interpreter = FakeInterpreter(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    event = interpreter.interpret(
        GestureWindow(
            clip_id="demo",
            sampled_frames=(),
            token_trace=(),
            video_path=None,
            candidate_labels=("COMPUTER", "SNOW"),
        )
    )

    assert event.gloss == ("COMPUTER",)
    assert event.sentence == "Computer."
    assert event.model_mode == "vlm"
    assert event.confidence == 0.88
