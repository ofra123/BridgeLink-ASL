"""Gradio Space entrypoint for BridgeLink ASL."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.clip_dataset import load_clip_dataset  # noqa: E402
from bridgelink_asl.cnn import CnnModelConfig, build_cnn_training_plan, describe_cnn_baseline  # noqa: E402
from bridgelink_asl.config import load_config  # noqa: E402
from bridgelink_asl.space_inference import result_to_markdown, run_space_poc  # noqa: E402


def predict_asl_clip(video: Any, mode: str) -> tuple[str, dict[str, Any]]:
    """Run the hosted proof-of-concept path."""

    config = load_config()
    result = run_space_poc(video, mode, vlm_model_id=config.vlm_model_id)
    return result_to_markdown(result), result


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


with gr.Blocks(title="BridgeLink ASL") as demo:
    gr.Markdown(
        """
        # BridgeLink ASL

        CNN vs VLM ASL sentence recognition demo. Record a short webcam clip
        or upload a clip, then run CNN, VLM, or Compare mode.

        The hosted Space runs an end-to-end proof of concept today. It analyzes
        the clip, produces CNN-style and VLM-style outputs, and keeps the same
        contracts that the trained CNN and Qwen runtime will use later.
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
        summary_output = gr.Markdown(label="Summary")
        output = gr.JSON(label="Detailed Result")
        run_button.click(predict_asl_clip, inputs=[video_input, mode_input], outputs=[summary_output, output])

    with gr.Tab("Webcam Only"):
        gr.Markdown(
            """
            This tab forces webcam recording only. Record a short sentence clip,
            stop recording, then run the same comparison pipeline.
            """
        )
        webcam_input = gr.Video(
            label="Record webcam signing clip",
            sources=["webcam"],
            format="mp4",
            include_audio=False,
        )
        webcam_mode = gr.Dropdown(["Compare", "CNN", "VLM"], value="Compare", label="Mode")
        webcam_button = gr.Button("Run Captured Webcam Clip")
        webcam_summary = gr.Markdown(label="Summary")
        webcam_output = gr.JSON(label="Detailed Result")
        webcam_button.click(
            predict_asl_clip,
            inputs=[webcam_input, webcam_mode],
            outputs=[webcam_summary, webcam_output],
        )

    with gr.Tab("Project Status"):
        gr.Markdown(
            """
            ## What runs on this Space now

            - Upload or record a short ASL clip.
            - Extract lightweight video features with OpenCV when available.
            - Produce a CNN-style sentence/gloss prediction.
            - Produce a VLM-style English sentence grounded by the CNN/token output.
            - Show agreement, confidence, latency, and limitations.

            ## What still requires project data/hardware

            - Omar's trained CNN artifact needs real sampled frames from Trey.
            - Frank's real Qwen2.5-VL-32B-AWQ inference needs GPU hardware and model setup.
            - Dalen's final benchmark wrapper will replace the current PoC scoring with true test-set metrics.
            """
        )

    with gr.Tab("CNN Plan"):
        plan_button = gr.Button("Show CNN Training Plan")
        plan_output = gr.JSON(label="CNN Plan")
        plan_button.click(cnn_plan, outputs=plan_output)


if __name__ == "__main__":
    demo.launch()
