"""Gradio Space entrypoint for BridgeLink ASL."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.clip_dataset import ClipDatasetRecord, load_clip_dataset  # noqa: E402
from bridgelink_asl.cnn import CnnModelConfig, build_cnn_training_plan, describe_cnn_baseline  # noqa: E402
from bridgelink_asl.config import load_config  # noqa: E402


def predict_asl_clip(video: str | None, mode: str) -> dict[str, Any]:
    """Run the hosted demo path.

    The Space intentionally starts in mock mode so the UI stays available before
    the large CNN/VLM artifacts are installed on Hugging Face hardware.
    """

    start_time = time.perf_counter()
    config = load_config()
    record = _load_reference_record(config.clip_manifest_path)
    selected_mode = mode.strip().lower()
    cnn_result = _mock_cnn_result(record) if selected_mode in {"cnn", "compare"} else {"status": "not_run"}
    vlm_result = _mock_vlm_result(record, config.vlm_model_id) if selected_mode in {"vlm", "compare"} else {"status": "not_run"}
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "mode": selected_mode,
        "uploaded_video": Path(video).name if video else None,
        "status": "mock_hosted_demo",
        "note": (
            "This Space is wired for the CNN vs VLM demo UI. Real CNN/VLM inference "
            "will be enabled after model artifacts and Hugging Face hardware are configured."
        ),
        "target_vlm_model": config.vlm_model_id,
        "cnn": cnn_result,
        "vlm": vlm_result,
        "reference_clip": {
            "clip_id": record.clip_id,
            "expected_gloss": list(record.gloss),
            "expected_english": record.english,
        },
        "latency_ms": latency_ms,
    }


def cnn_plan() -> dict[str, Any]:
    """Show Omar's CNN baseline plan in the hosted UI."""

    config = load_config()
    records = load_clip_dataset(config.clip_manifest_path)
    cnn_config = CnnModelConfig(
        frame_count=config.cnn_frame_count,
        image_size=config.cnn_image_size,
        batch_size=config.cnn_batch_size,
        epochs=config.cnn_epochs,
        model_path=config.cnn_model_path,
        manifest_path=config.clip_manifest_path,
    )
    plan = build_cnn_training_plan(records, cnn_config)
    return {
        "cnn": describe_cnn_baseline(cnn_config, num_classes=len(plan.labels)),
        "plan": plan.as_dict(),
    }


def _load_reference_record(manifest_path: Path) -> ClipDatasetRecord:
    records = load_clip_dataset(manifest_path)
    if not records:
        raise ValueError(f"No clip records found in {manifest_path}.")
    return records[0]


def _mock_cnn_result(record: ClipDatasetRecord) -> dict[str, Any]:
    return {
        "status": "mock",
        "prediction": record.label,
        "confidence": 0.82,
        "explanation": "Placeholder CNN output using the sample manifest target.",
    }


def _mock_vlm_result(record: ClipDatasetRecord, model_id: str) -> dict[str, Any]:
    return {
        "status": "mock",
        "model_id": model_id,
        "sentence": record.english,
        "confidence": 0.78,
        "needs_clarification": False,
        "explanation": "Placeholder VLM output for the configured local/hosted Qwen target.",
    }


with gr.Blocks(title="BridgeLink ASL") as demo:
    gr.Markdown(
        """
        # BridgeLink ASL

        CNN vs VLM ASL sentence recognition demo. Record a short webcam clip
        or upload a clip, then run CNN, VLM, or Compare mode.

        The hosted Space starts in mock mode so the UI stays usable while the
        CNN artifact and Qwen VLM runtime are being prepared. For the 32B VLM,
        short captured clips are the practical "live" path; true continuous
        streaming is a stretch goal because large VLM inference is slow.
        """
    )
    with gr.Tab("Run Demo"):
        gr.Markdown(
            """
            Use the webcam option to record a 2-5 second signing clip. This is
            the recommended hosted-demo flow for Hugging Face Spaces.
            """
        )
        video_input = gr.Video(
            label="Upload or record a short ASL clip",
            sources=["upload", "webcam"],
            format="mp4",
            include_audio=False,
        )
        mode_input = gr.Dropdown(["Compare", "CNN", "VLM"], value="Compare", label="Mode")
        run_button = gr.Button("Run BridgeLink ASL")
        output = gr.JSON(label="Result")
        run_button.click(predict_asl_clip, inputs=[video_input, mode_input], outputs=output)

    with gr.Tab("Webcam Only"):
        gr.Markdown(
            """
            This tab forces webcam recording only. Record a short sentence clip,
            stop recording, then run the same comparison pipeline.
            """
        )
        webcam_input = gr.Video(
            label="Record webcam signing clip",
            sources="webcam",
            format="mp4",
            include_audio=False,
        )
        webcam_mode = gr.Dropdown(["Compare", "CNN", "VLM"], value="Compare", label="Mode")
        webcam_button = gr.Button("Run Captured Webcam Clip")
        webcam_output = gr.JSON(label="Result")
        webcam_button.click(predict_asl_clip, inputs=[webcam_input, webcam_mode], outputs=webcam_output)

    with gr.Tab("CNN Plan"):
        plan_button = gr.Button("Show CNN Training Plan")
        plan_output = gr.JSON(label="CNN Plan")
        plan_button.click(cnn_plan, outputs=plan_output)


if __name__ == "__main__":
    demo.launch()
