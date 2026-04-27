"""BridgeLink ASL Hugging Face Space entrypoint.

This Space now exposes two model paths:
- A landmark-based WLASL live/demo classifier.
- A How2Sign sentence-level 3D CNN for uploaded clips.

Set HF_MODEL_REPO / HF_MODEL_FILENAME for the landmark model and
HF_SENTENCE_MODEL_REPO / HF_SENTENCE_MODEL_FILENAME for the How2Sign model.
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
    process_frame_for_landmarks,
    draw_tracking_overlay,
    extract_landmarks_from_video,
)
from bridgelink_asl.sentence_inference import (  # noqa: E402
    SentenceClipRuntime,
    extract_clip_volume,
    load_sentence_runtime,
)

# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------

MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "cnn_landmark_wlasl25_best.pt").strip()
LOCAL_WEIGHTS = PROJECT_ROOT / "models" / MODEL_FILENAME
SENTENCE_MODEL_REPO = os.environ.get("HF_SENTENCE_MODEL_REPO", "").strip()
SENTENCE_MODEL_FILENAME = os.environ.get("HF_SENTENCE_MODEL_FILENAME", "cnn-3d-sentence-top25-normalized.keras").strip()
LOCAL_SENTENCE_WEIGHTS = PROJECT_ROOT / "models" / SENTENCE_MODEL_FILENAME
SEQ_LEN = 32
STRIDE = 4                     # run inference every STRIDE frames
MIN_CONFIDENCE = float(os.environ.get("BRIDGELINK_MIN_CONFIDENCE", "0.55"))
SENTENCE_MIN_CONFIDENCE = float(os.environ.get("BRIDGELINK_SENTENCE_MIN_CONFIDENCE", "0.55"))
STABILITY_K = 2                # require K consecutive same predictions before emitting
SENTENCE_STREAM_STRIDE = int(os.environ.get("BRIDGELINK_SENTENCE_STREAM_STRIDE", "2"))
SENTENCE_STABILITY_K = int(os.environ.get("BRIDGELINK_SENTENCE_STABILITY_K", "2"))

RUNTIME: SignLanguageRuntime | None = None
RUNTIME_ERROR: str | None = None
SENTENCE_RUNTIME: SentenceClipRuntime | None = None
SENTENCE_RUNTIME_ERROR: str | None = None

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

try:
    SENTENCE_RUNTIME = load_sentence_runtime(
        local_model_path=LOCAL_SENTENCE_WEIGHTS,
        hf_repo=SENTENCE_MODEL_REPO or None,
        hf_model_filename=SENTENCE_MODEL_FILENAME,
    )
    print(f"[bridgelink] loaded sentence cnn with {len(SENTENCE_RUNTIME.labels)} classes")
except Exception as exc:  # keep the Space bootable even if weights are missing
    SENTENCE_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[bridgelink] WARNING: sentence cnn not loaded — {SENTENCE_RUNTIME_ERROR}")


def _require_runtime() -> SignLanguageRuntime:
    if RUNTIME is None:
        raise gr.Error(
            "Model weights are not available on this Space. "
            "Set the HF_MODEL_REPO env var or upload "
            "models/cnn_landmark_wlasl25_best.pt to the repo."
        )
    return RUNTIME


def _require_sentence_runtime() -> SentenceClipRuntime:
    if SENTENCE_RUNTIME is None:
        raise gr.Error(
            "How2Sign sentence CNN weights are not available on this Space. "
            "Set HF_SENTENCE_MODEL_REPO or upload "
            "models/cnn-3d-sentence-top25-normalized.keras and its labels file to the repo."
        )
    return SENTENCE_RUNTIME


# ---------------------------------------------------------------------------
# Live webcam streaming handler
# ---------------------------------------------------------------------------

def init_live_state() -> dict[str, Any]:
    sentence_window = SENTENCE_RUNTIME.frame_count if SENTENCE_RUNTIME is not None else 16
    return {
        "buffer": deque(maxlen=SEQ_LEN),
        "sentence_buffer": deque(maxlen=sentence_window),
        "frame_idx": 0,
        "last_label": None,
        "candidate_label": None,
        "candidate_confidence": None,
        "candidate_top5": [],
        "stable_count": 0,
        "caption": "",
        "history": [],
        "last_inference_ms": 0.0,
        "sentence_last_label": None,
        "sentence_candidate_label": None,
        "sentence_candidate_confidence": None,
        "sentence_candidate_top5": [],
        "sentence_stable_count": 0,
        "sentence_caption": "",
        "sentence_history": [],
        "sentence_last_inference_ms": 0.0,
    }


def on_live_frame(frame: np.ndarray, state: dict[str, Any]):
    """Streaming handler — called on every webcam frame from Gradio."""
    if state is None:
        state = init_live_state()
    if frame is None:
        return None, _status_markdown(state), _sentence_status_markdown(state), state

    runtime = _require_runtime()

    # Extract landmarks for this frame and append to the rolling buffer.
    lm, tracking_result = process_frame_for_landmarks(frame)
    state["buffer"].append(lm)
    state["frame_idx"] += 1
    _update_sentence_live_state(frame, state)

    # Only run the model every STRIDE frames, and only once the buffer is full.
    if (
        len(state["buffer"]) == SEQ_LEN
        and state["frame_idx"] % STRIDE == 0
    ):
        t0 = time.perf_counter()
        sequence = np.stack(list(state["buffer"]))   # (32, 225)
        label, confidence, top5 = runtime.predict(sequence)
        state["last_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        state["candidate_confidence"] = confidence

        if confidence >= MIN_CONFIDENCE:
            state["candidate_label"] = label
            state["candidate_top5"] = top5
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
        else:
            state["candidate_label"] = None
            state["candidate_top5"] = []
            state["stable_count"] = 0

    tracking_overlay = draw_tracking_overlay(
        frame,
        tracking_result,
        label=state.get("candidate_label"),
        confidence=state.get("candidate_confidence"),
        top5=state.get("candidate_top5"),
        buffer_size=len(state["buffer"]),
    )
    return tracking_overlay, _status_markdown(state), _sentence_status_markdown(state), state


def _update_sentence_live_state(frame: np.ndarray, state: dict[str, Any]) -> None:
    if SENTENCE_RUNTIME is None:
        return

    import cv2

    resized = cv2.resize(
        frame,
        (SENTENCE_RUNTIME.image_size, SENTENCE_RUNTIME.image_size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    state["sentence_buffer"].append(resized)

    if (
        len(state["sentence_buffer"]) != SENTENCE_RUNTIME.frame_count
        or state["frame_idx"] % SENTENCE_STREAM_STRIDE != 0
    ):
        return

    clip_volume = np.stack(list(state["sentence_buffer"]), axis=0)
    t0 = time.perf_counter()
    label, confidence, top5 = SENTENCE_RUNTIME.predict_clip(clip_volume)
    state["sentence_last_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    state["sentence_candidate_confidence"] = confidence

    if confidence < SENTENCE_MIN_CONFIDENCE:
        state["sentence_candidate_label"] = None
        state["sentence_candidate_top5"] = []
        state["sentence_stable_count"] = 0
        return

    state["sentence_candidate_label"] = label
    state["sentence_candidate_top5"] = top5
    if label == state["sentence_last_label"]:
        state["sentence_stable_count"] += 1
    else:
        state["sentence_last_label"] = label
        state["sentence_stable_count"] = 1

    if state["sentence_stable_count"] < SENTENCE_STABILITY_K:
        return

    last_emitted = state["sentence_history"][-1] if state["sentence_history"] else None
    if label != last_emitted:
        state["sentence_history"].append(label)
        state["sentence_history"] = state["sentence_history"][-4:]
    state["sentence_caption"] = label


def _format_caption(history: list[str]) -> str:
    if not history:
        return "_Start signing..._"
    words = [w.replace("_", " ").title() for w in history]
    return "**" + " · ".join(words) + "**"


def _status_markdown(state: dict[str, Any]) -> str:
    caption = state.get("caption") or "_Start signing..._"
    last_label = state.get("candidate_label") or "--"
    confidence = state.get("candidate_confidence")
    confidence_text = f"{confidence:.1%}" if confidence is not None else "--"
    infer = state.get("last_inference_ms", 0.0)
    buf = len(state["buffer"]) if state.get("buffer") is not None else 0
    return (
        f"### Caption\n\n{caption}\n\n"
        f"- Current candidate: **{last_label}**\n"
        f"- Candidate confidence: **{confidence_text}**\n"
        f"- Buffer: {buf}/{SEQ_LEN} frames\n"
        f"- Last inference: {infer} ms"
    )


def _format_caption(history: list[str]) -> str:
    if not history:
        return "_Waiting for a confident sign..._"
    words = [w.replace("_", " ").title() for w in history]
    return "**" + " / ".join(words) + "**"


def _status_markdown(state: dict[str, Any]) -> str:
    caption = state.get("caption") or "_Waiting for a confident sign..._"
    last_label = state.get("candidate_label") or "Waiting"
    confidence = state.get("candidate_confidence")
    confidence_text = f"{confidence:.1%}" if confidence is not None else "--"
    infer = state.get("last_inference_ms", 0.0)
    buf = len(state["buffer"]) if state.get("buffer") is not None else 0
    return (
        f"### Caption\n\n{caption}\n\n"
        f"- Confirmed candidate: **{last_label}**\n"
        f"- Confidence gate: **{MIN_CONFIDENCE:.0%}**\n"
        f"- Latest confidence: **{confidence_text}**\n"
        f"- Buffer: {buf}/{SEQ_LEN} frames\n"
        f"- Last inference: {infer} ms"
    )


def _sentence_status_markdown(state: dict[str, Any]) -> str:
    if SENTENCE_RUNTIME is None:
        return "### Sentence Watch\n\n_How2Sign sentence CNN is not loaded on this Space._"

    caption = state.get("sentence_caption") or "_Waiting for a confident sentence..._"
    last_label = state.get("sentence_candidate_label") or "Waiting"
    confidence = state.get("sentence_candidate_confidence")
    confidence_text = f"{confidence:.1%}" if confidence is not None else "--"
    infer = state.get("sentence_last_inference_ms", 0.0)
    buf = len(state["sentence_buffer"]) if state.get("sentence_buffer") is not None else 0
    return (
        f"### Sentence Watch\n\n{caption}\n\n"
        "- Closed vocabulary: **How2Sign repeated sentences**\n"
        f"- Current sentence: **{last_label}**\n"
        f"- Confidence gate: **{SENTENCE_MIN_CONFIDENCE:.0%}**\n"
        f"- Latest confidence: **{confidence_text}**\n"
        f"- Buffer: {buf}/{SENTENCE_RUNTIME.frame_count} frames\n"
        f"- Last inference: {infer} ms"
    )


def reset_live() -> tuple[dict[str, Any], str, str]:
    state = init_live_state()
    return state, _status_markdown(state), _sentence_status_markdown(state)


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

    if confidence < MIN_CONFIDENCE:
        details = {
            "raw_label": label,
            "accepted_label": None,
            "confidence": confidence,
            "confidence_gate": MIN_CONFIDENCE,
            "top5": [{"label": n, "score": s} for n, s in top5],
            "extract_ms": extract_ms,
            "inference_ms": infer_ms,
            "sequence_shape": list(sequence.shape),
        }
        return (
            "## No Confident Sign Yet\n\n"
            f"Highest confidence: **{confidence:.1%}**\n\n"
            f"Confidence gate: **{MIN_CONFIDENCE:.0%}**",
            details,
        )

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
        "raw_label": label,
        "accepted_label": label,
        "confidence": confidence,
        "confidence_gate": MIN_CONFIDENCE,
        "top5": [{"label": n, "score": s} for n, s in top5],
        "extract_ms": extract_ms,
        "inference_ms": infer_ms,
        "sequence_shape": list(sequence.shape),
    }
    return "\n".join(lines), details


def classify_sentence_clip(video_path: str | None) -> tuple[str, dict[str, Any]]:
    if not video_path:
        return "Please upload or record a clip first.", {}

    runtime = _require_sentence_runtime()

    t0 = time.perf_counter()
    clip_volume, clip_meta = extract_clip_volume(
        video_path,
        frame_count=runtime.frame_count,
        image_size=runtime.image_size,
    )
    extract_ms = round((time.perf_counter() - t0) * 1000, 1)

    if clip_volume is None:
        return "Could not decode the clip for the sentence CNN.", dict(clip_meta)

    t0 = time.perf_counter()
    retrieval_mode = runtime.embedding_model is not None and runtime.embedding_index is not None
    retrieval_meta: dict[str, Any] = {}
    if retrieval_mode:
        label, confidence, _, retrieval_meta = runtime.predict_clip_retrieval(clip_volume)
    else:
        label, confidence, _ = runtime.predict_clip(clip_volume)
    infer_ms = round((time.perf_counter() - t0) * 1000, 1)

    if retrieval_mode:
        support_split = getattr(runtime.embedding_index, "support_split", None) if runtime.embedding_index is not None else None
        details = {
            "raw_label": label,
            "accepted_label": label,
            "match_similarity": confidence,
            "extract_ms": extract_ms,
            "inference_ms": infer_ms,
            "clip_shape": list(clip_volume.shape),
            "frame_count": runtime.frame_count,
            "image_size": runtime.image_size,
            "runtime_model_path": runtime.model_path,
            "sentence_inference_mode": runtime.inference_mode,
            "index_path": getattr(runtime.embedding_index, "index_path", None),
            "source_manifest": retrieval_meta.get("source_manifest"),
            "support_split": support_split,
            **clip_meta,
        }
        lines = [
            "## Best Sentence Match",
            "",
            f"# {label}",
            f"Nearest support-clip similarity: **{confidence:.3f}**",
        ]
        lines += [
            "",
            f"Frames sampled: {clip_meta['sampled_frames']} from {clip_meta['source_frames']} decoded frames",
            f"Preprocess: {extract_ms} ms / Inference: {infer_ms} ms",
            f"Inference mode: {runtime.inference_mode}",
            f"Model path: {runtime.model_path}",
        ]
        if runtime.embedding_index is not None and runtime.embedding_index.index_path:
            lines.append(f"Index path: {runtime.embedding_index.index_path}")
        if support_split:
            lines.append(f"Support set: {support_split}")
        lines += [
            "",
            "Note: this result comes from the 3D CNN embedding space matched against the repeated-sentence reference set.",
        ]
        return "\n".join(lines), details

    if confidence < SENTENCE_MIN_CONFIDENCE:
        details = {
            "raw_label": label,
            "accepted_label": None,
            "highest_confidence": confidence,
            "confidence_gate": SENTENCE_MIN_CONFIDENCE,
            "extract_ms": extract_ms,
            "inference_ms": infer_ms,
            "clip_shape": list(clip_volume.shape),
            "frame_count": runtime.frame_count,
            "image_size": runtime.image_size,
            "runtime_model_path": runtime.model_path,
            "sentence_inference_mode": runtime.inference_mode,
            **clip_meta,
        }
        lines = [
            "## No Confident Sentence Yet",
            "",
            "The clip was processed, but the model did not clear the display threshold.",
            "",
            f"Highest confidence: **{confidence:.1%}**",
            f"Confidence gate: **{SENTENCE_MIN_CONFIDENCE:.0%}**",
            "",
            f"Frames sampled: {clip_meta['sampled_frames']} from {clip_meta['source_frames']} decoded frames",
            f"Preprocess: {extract_ms} ms / Inference: {infer_ms} ms",
            f"Inference mode: {runtime.inference_mode}",
            f"Model path: {runtime.model_path}",
        ]
        return "\n".join(lines), details

    lines = [
        "## Confident Sentence",
        "",
        f"# {label}",
        f"Confidence: **{confidence:.1%}**",
        "",
    ]
    lines += [
        "",
        f"Frames sampled: {clip_meta['sampled_frames']} from {clip_meta['source_frames']} decoded frames",
        f"Preprocess: {extract_ms} ms / Inference: {infer_ms} ms",
        "",
        "Note: this is a closed-vocabulary How2Sign sentence model trained on repeated-sentence subsets.",
    ]

    details = {
        "raw_label": label,
        "accepted_label": label,
        "confidence": confidence,
        "confidence_gate": SENTENCE_MIN_CONFIDENCE,
        "extract_ms": extract_ms,
        "inference_ms": infer_ms,
        "clip_shape": list(clip_volume.shape),
        "frame_count": runtime.frame_count,
        "image_size": runtime.image_size,
        "runtime_model_path": runtime.model_path,
        "sentence_inference_mode": runtime.inference_mode,
        **clip_meta,
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

LANDMARK_STATUS_BANNER = (
    f"WLASL landmark model: **loaded — {len(RUNTIME.labels)} classes**"
    if RUNTIME is not None
    else f"WLASL landmark model: **NOT LOADED** ({RUNTIME_ERROR})"
)

SENTENCE_STATUS_BANNER = (
    f"How2Sign sentence CNN: **loaded — {len(SENTENCE_RUNTIME.labels)} classes**"
    if SENTENCE_RUNTIME is not None
    else f"How2Sign sentence CNN: **NOT LOADED** ({SENTENCE_RUNTIME_ERROR})"
)


with gr.Blocks(title="BridgeLink ASL") as demo:
    gr.Markdown(
        f"""
        # BridgeLink ASL

        Real-time American Sign Language recognition demos for both
        landmark-based isolated-sign classification and RGB clip-based
        sentence classification.

        - {LANDMARK_STATUS_BANNER}
        - {SENTENCE_STATUS_BANNER}
        """
    )

    with gr.Tab("Live Webcam"):
        gr.Markdown(
            "Stream your webcam to run both live models at once. "
            "The landmark CNN drives the fast sign caption, while the How2Sign "
            "sentence CNN watches the RGB clip buffer for a confident repeated-sentence match. "
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
                tracking_view = gr.Image(
                    type="numpy",
                    label="MediaPipe tracking overlay",
                    interactive=False,
                )
            with gr.Column(scale=1):
                live_status = gr.Markdown(_status_markdown(init_live_state()))
                sentence_live_status = gr.Markdown(_sentence_status_markdown(init_live_state()))
                reset_btn = gr.Button("Reset caption")
        webcam.stream(
            on_live_frame,
            inputs=[webcam, live_state],
            outputs=[tracking_view, live_status, sentence_live_status, live_state],
            show_progress="hidden",
            stream_every=0.2,
        )
        reset_btn.click(reset_live, outputs=[live_state, live_status, sentence_live_status])

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

    with gr.Tab("How2Sign Sentence CNN"):
        gr.Markdown(
            "Upload a short How2Sign-style RGB clip to run the closed-vocabulary "
            "sentence classifier. This model predicts one sentence from the "
            "repeated-sentence subset used in the 3D CNN experiments."
        )
        sentence_clip_input = gr.Video(
            sources=["upload", "webcam"],
            format="mp4",
            include_audio=False,
            label="How2Sign sentence clip",
        )
        sentence_button = gr.Button("Classify sentence clip", variant="primary")
        sentence_summary = gr.Markdown()
        sentence_details = gr.JSON(label="Sentence CNN details")
        sentence_button.click(
            classify_sentence_clip,
            inputs=[sentence_clip_input],
            outputs=[sentence_summary, sentence_details],
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
            5. **Sentence baseline** — sampled RGB clips from How2Sign passed through a 3D CNN that predicts one repeated sentence label

            ## Dataset

            [WLASL-100](https://dxli94.github.io/WLASL/) — the 100 most frequent glosses
            from the Word-Level ASL video dataset. Distributed under the Computational
            Use of Data Agreement (C-UDA).

            [How2Sign](https://how2sign.github.io/) — RGB sign-language sentence clips.
            The current sentence CNN demo is limited to a repeated-sentence subset so the
            classifier has enough examples per class.

            ## Limitations

            - Trained on WLASL-100 only — vocabulary is limited to 100 glosses.
            - Single-signer generalization depends on the diversity of the training split.
            - Continuous sentence translation is out of scope; the model classifies
              isolated signs from short windows.
            - The How2Sign sentence demo is closed-vocabulary and only supports the
              repeated sentence classes used during training.
            """
        )


if __name__ == "__main__":
    demo.launch()
