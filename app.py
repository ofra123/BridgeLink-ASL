"""BridgeLink ASL — Hugging Face Space entrypoint.

Loads a trained landmark CNN by default, with Transformer checkpoints supported
as an extra-credit/attention model, and runs two demo modes:
- Live webcam streaming (frame-by-frame sliding window prediction).
- Uploaded / recorded clip classification.

Set the HF_MODEL_REPO env var (e.g. "your-username/bridgelink-asl-wlasl100")
to auto-download weights from the Hugging Face Hub on Space startup. If unset,
the app looks for models/cnn_landmark_best.pt in the repo root, then falls back
to models/sign_transformer_best.pt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.inference import (  # noqa: E402
    SignLanguageRuntime,
    load_runtime,
    extract_landmarks_from_frame,
    extract_landmarks_from_video,
)

# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------

MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "cnn_landmark_best.pt").strip()
LOCAL_WEIGHTS = PROJECT_ROOT / "models" / MODEL_FILENAME
SEQ_LEN = 32
STRIDE = 4                     # run inference every STRIDE frames
MIN_CONFIDENCE = 0.35          # below this, show nothing
STABILITY_K = 2                # require K consecutive same predictions before emitting

RUNTIME: SignLanguageRuntime | None = None
RUNTIME_ERROR: str | None = None

try:
    RUNTIME = load_runtime(
        local_path=LOCAL_WEIGHTS,
        hf_repo=MODEL_REPO or None,
        hf_filename=MODEL_FILENAME,
    )
    print(f"[bridgelink] loaded model with {len(RUNTIME.labels)} classes")
except Exception as exc:  # keep the Space bootable even if weights are missing
    RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[bridgelink] WARNING: model not loaded — {RUNTIME_ERROR}")


def _require_runtime() -> SignLanguageRuntime:
    if RUNTIME is None:
        raise gr.Error(
            "Model weights are not available on this Space. "
            "Set the HF_MODEL_REPO env var or upload "
            "models/cnn_landmark_best.pt to the repo."
        )
    return RUNTIME


# ---------------------------------------------------------------------------
# Live webcam streaming handler
# ---------------------------------------------------------------------------

def init_live_state() -> dict[str, Any]:
    return {
        "buffer": deque(maxlen=SEQ_LEN),
        "frame_idx": 0,
        "last_label": None,
        "stable_count": 0,
        "caption": "",
        "history": [],
        "last_inference_ms": 0.0,
    }


def on_live_frame(frame: np.ndarray, state: dict[str, Any]):
    """Streaming handler — called on every webcam frame from Gradio."""
    if state is None:
        state = init_live_state()
    if frame is None:
        return frame, state["caption"], state

    runtime = _require_runtime()

    # Extract landmarks for this frame and append to the rolling buffer.
    lm = extract_landmarks_from_frame(frame)
    state["buffer"].append(lm)
    state["frame_idx"] += 1

    # Only run the model every STRIDE frames, and only once the buffer is full.
    if (
        len(state["buffer"]) == SEQ_LEN
        and state["frame_idx"] % STRIDE == 0
    ):
        t0 = time.perf_counter()
        sequence = np.stack(list(state["buffer"]))   # (32, 225)
        label, confidence, top5 = runtime.predict(sequence)
        state["last_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if confidence >= MIN_CONFIDENCE:
            if label == state["last_label"]:
                state["stable_count"] += 1
            else:
                state["last_label"] = label
                state["stable_count"] = 1

            # Emit the sign only once it has been stable for K consecutive runs
            # AND is different from the most recently emitted caption token.
            if state["stable_count"] >= STABILITY_K:
                last_emitted = state["history"][-1] if state["history"] else None
                if label != last_emitted:
                    state["history"].append(label)
                    state["history"] = state["history"][-12:]  # keep last 12
                    state["caption"] = _format_caption(state["history"])

    return frame, _status_markdown(state), state


def _format_caption(history: list[str]) -> str:
    if not history:
        return "_Start signing..._"
    words = [w.replace("_", " ").title() for w in history]
    return "**" + " · ".join(words) + "**"


def _status_markdown(state: dict[str, Any]) -> str:
    caption = state.get("caption") or "_Start signing..._"
    last_label = state.get("last_label") or "—"
    infer = state.get("last_inference_ms", 0.0)
    buf = len(state["buffer"]) if state.get("buffer") is not None else 0
    return (
        f"### Caption\n\n{caption}\n\n"
        f"- Current candidate: **{last_label}**\n"
        f"- Buffer: {buf}/{SEQ_LEN} frames\n"
        f"- Last inference: {infer} ms"
    )


def reset_live() -> tuple[dict[str, Any], str]:
    state = init_live_state()
    return state, _status_markdown(state)


# ---------------------------------------------------------------------------
# Uploaded / recorded clip handler
# ---------------------------------------------------------------------------

def classify_clip(video_path: str | None) -> tuple[str, dict[str, Any]]:
    if not video_path:
        return "Please upload or record a clip first.", {}
    runtime = _require_runtime()

    t0 = time.perf_counter()
    sequence = extract_landmarks_from_video(video_path, seq_len=SEQ_LEN)
    extract_ms = round((time.perf_counter() - t0) * 1000, 1)

    if sequence is None:
        return "Could not read the video file.", {}

    t0 = time.perf_counter()
    label, confidence, top5 = runtime.predict(sequence)
    infer_ms = round((time.perf_counter() - t0) * 1000, 1)

    lines = [
        f"## Prediction: **{label.replace('_', ' ').title()}**",
        f"Confidence: **{confidence:.1%}**",
        "",
        "### Top 5",
    ]
    for i, (name, score) in enumerate(top5, start=1):
        lines.append(f"{i}. {name.replace('_', ' ').title()} — {score:.1%}")
    lines += ["", f"Landmark extraction: {extract_ms} ms · Inference: {infer_ms} ms"]

    details = {
        "label": label,
        "confidence": confidence,
        "top5": [{"label": n, "score": s} for n, s in top5],
        "extract_ms": extract_ms,
        "inference_ms": infer_ms,
        "sequence_shape": list(sequence.shape),
    }
    return "\n".join(lines), details


# ---------------------------------------------------------------------------
# Dataset / results tabs (read static artifacts exported by the notebook)
# ---------------------------------------------------------------------------

def load_metrics() -> dict[str, Any]:
    candidates = [
        PROJECT_ROOT / "results" / "cnn_metrics.json",
        PROJECT_ROOT / "results" / "metrics.json",
        PROJECT_ROOT / "models" / "metrics.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    return {"status": "metrics.json not found — run the training notebook and upload it."}


def results_markdown() -> str:
    m = load_metrics()
    if "status" in m:
        return m["status"]
    return (
        "## WLASL-100 results\n\n"
        f"- Model: **{m.get('model', 'landmark model')}**\n"
        f"- Test top-1: **{m.get('test_top1', 0):.1%}**\n"
        f"- Test top-5: **{m.get('test_top5', 0):.1%}**\n"
        f"- Best val top-1: {m.get('val_top1_best', 0):.1%} "
        f"(epoch {m.get('val_top1_best_epoch', '?')})\n"
        f"- Classes: {m.get('num_classes', '?')}\n"
        f"- Train / val / test: "
        f"{m.get('train_samples', '?')} / {m.get('val_samples', '?')} / {m.get('test_samples', '?')}\n"
        f"- Model params: {m.get('model_params_M', 0):.2f}M\n"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

STATUS_BANNER = (
    f"Model: **loaded — {len(RUNTIME.labels)} classes**"
    if RUNTIME is not None
    else f"Model: **NOT LOADED** ({RUNTIME_ERROR})"
)


with gr.Blocks(title="BridgeLink ASL") as demo:
    gr.Markdown(
        f"""
        # BridgeLink ASL

        Real-time American Sign Language word recognition using MediaPipe
        landmarks and a lightweight CNN classifier trained on WLASL-100.
        Transformer checkpoints are also supported as an attention-based
        extension.

        {STATUS_BANNER}
        """
    )

    with gr.Tab("Live Webcam"):
        gr.Markdown(
            "Stream your webcam. The model runs on a rolling 32-frame window "
            "and emits a sign when it is confident and stable. "
            "For best results: good lighting, plain background, upper body in frame."
        )
        live_state = gr.State(init_live_state())
        with gr.Row():
            with gr.Column(scale=2):
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                )
            with gr.Column(scale=1):
                live_status = gr.Markdown(_status_markdown(init_live_state()))
                reset_btn = gr.Button("Reset caption")
        webcam.stream(
            on_live_frame,
            inputs=[webcam, live_state],
            outputs=[webcam, live_status, live_state],
            show_progress="hidden",
        )
        reset_btn.click(reset_live, outputs=[live_state, live_status])

    with gr.Tab("Upload / Record Clip"):
        gr.Markdown(
            "Upload an mp4 or record a short (2–5 s) clip. Use this tab as a "
            "reliable backup if the live stream is laggy."
        )
        clip_input = gr.Video(
            sources=["upload", "webcam"],
            format="mp4",
            include_audio=False,
            label="ASL clip",
        )
        clip_button = gr.Button("Classify clip", variant="primary")
        clip_summary = gr.Markdown()
        clip_details = gr.JSON(label="Details")
        clip_button.click(
            classify_clip,
            inputs=[clip_input],
            outputs=[clip_summary, clip_details],
        )

    with gr.Tab("Results"):
        results_md = gr.Markdown(results_markdown())
        gr.Markdown(
            "Confusion matrix, training curves, and classification report are "
            "exported by the training notebook into the `results/` folder and "
            "embedded in the CVPR-style report."
        )

    with gr.Tab("About"):
        gr.Markdown(
            """
            ## Method

            1. **Landmark extraction** — MediaPipe Holistic (21 left-hand + 21 right-hand + 33 pose landmarks × 3 coords = 225 dims per frame)
            2. **Sequence model** — 1D landmark CNN over the 32-frame temporal sequence; optional Transformer extension for attention comparison
            3. **Training** — WLASL-100, AdamW + cosine schedule, label smoothing, temporal + spatial augmentation
            4. **Inference** — rolling 32-frame buffer, stride 4, confidence threshold 0.35, stability filter of 2 consecutive frames before emission

            ## Dataset

            [WLASL-100](https://dxli94.github.io/WLASL/) — the 100 most frequent glosses
            from the Word-Level ASL video dataset. Distributed under the Computational
            Use of Data Agreement (C-UDA).

            ## Limitations

            - Trained on WLASL-100 only — vocabulary is limited to 100 glosses.
            - Single-signer generalization depends on the diversity of the training split.
            - Continuous sentence translation is out of scope; the model classifies
              isolated signs from short windows.
            """
        )


if __name__ == "__main__":
    demo.launch()
