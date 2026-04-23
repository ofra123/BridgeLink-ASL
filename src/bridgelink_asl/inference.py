"""Runtime inference module for BridgeLink ASL.

Kept separate from the training code so the Hugging Face Space has a minimal
dependency footprint: torch, mediapipe, opencv, numpy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Feature constants (must match the training notebook)
# ---------------------------------------------------------------------------

SEQ_LEN_DEFAULT = 32
FEAT_DIM = 21 * 3 + 21 * 3 + 33 * 3  # 225

# ---------------------------------------------------------------------------
# Model definition — identical architecture to the training notebook
# ---------------------------------------------------------------------------

def _build_transformer_model(config: dict):
    import torch
    import torch.nn as nn

    class SignTransformer(nn.Module):
        def __init__(self, num_classes, d_model=192, nhead=4, layers=4,
                     seq_len=32, feat_dim=225, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(feat_dim, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.size(0)
            h = self.input_proj(x)
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1) + self.pos_embed
            h = self.encoder(h)
            h = self.norm(h[:, 0])
            return self.head(h)

    return SignTransformer(
        num_classes=config["num_classes"],
        d_model=config.get("d_model", 192),
        nhead=config.get("nhead", 4),
        layers=config.get("layers", 4),
        seq_len=config.get("seq_len", 32),
        feat_dim=config.get("feat_dim", 225),
        dropout=config.get("dropout", 0.3),
    )


def _build_cnn_model(config: dict):
    import torch
    import torch.nn as nn

    class LandmarkCNN(nn.Module):
        def __init__(
            self,
            num_classes: int,
            feat_dim: int = 225,
            channels: tuple[int, int, int] = (128, 128, 256),
            dropout: float = 0.35,
        ):
            super().__init__()
            c1, c2, c3 = channels
            self.features = nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Conv1d(feat_dim, c1, kernel_size=3, padding=1),
                nn.BatchNorm1d(c1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(c1, c2, kernel_size=3, padding=1),
                nn.BatchNorm1d(c2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(c2, c3, kernel_size=3, padding=1),
                nn.BatchNorm1d(c3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(c3, num_classes)

        def forward(self, x):
            h = x.transpose(1, 2)
            h = self.features(h).squeeze(-1)
            return self.head(h)

    return LandmarkCNN(
        num_classes=config["num_classes"],
        feat_dim=config.get("feat_dim", 225),
        channels=tuple(config.get("channels", (128, 128, 256))),
        dropout=config.get("dropout", 0.35),
    )


def _build_model(config: dict):
    """Build the runtime model declared by a checkpoint config."""

    model_type = str(config.get("model_type", "transformer")).lower()
    if model_type in {"cnn", "landmark_cnn", "temporal_cnn"}:
        return _build_cnn_model(config)
    return _build_transformer_model(config)


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------

@dataclass
class SignLanguageRuntime:
    model: object            # torch.nn.Module
    labels: list[str]        # index -> gloss string
    config: dict
    device: str

    def predict(self, sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """Run inference on a single (seq_len, 225) landmark sequence."""
        import torch
        with torch.no_grad():
            x = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        top5_idx = probs.argsort()[::-1][:5]
        top5 = [(self.labels[int(i)], float(probs[int(i)])) for i in top5_idx]
        best_idx = int(probs.argmax())
        return self.labels[best_idx], float(probs[best_idx]), top5


def load_runtime(
    local_path: Path | str = "models/cnn_landmark_wlasl25_best.pt",
    hf_repo: Optional[str] = None,
    hf_filename: str | None = None,
) -> SignLanguageRuntime:
    """Load model weights from a local path, with optional HF Hub fallback."""
    import torch

    local_path = Path(local_path)
    filename = hf_filename or local_path.name

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=hf_repo, filename=filename)
            local_path = Path(downloaded)
        except Exception as exc:
            print(f"[bridgelink] HF Hub download failed ({exc}); falling back to local path")

    if not local_path.exists():
        fallback_names = ["cnn_landmark_best.pt", "sign_transformer_best.pt"]
        for fallback_name in fallback_names:
            fallback_path = local_path.parent / fallback_name
            if fallback_path.exists():
                local_path = fallback_path
                break
        else:
            expected = f"{filename}, cnn_landmark_best.pt, or sign_transformer_best.pt"
            raise FileNotFoundError(
                f"Model weights not found at {local_path}. "
                f"Train the model with the Colab notebook and place the .pt file there, "
                f"or set HF_MODEL_REPO to a HF model repo containing {expected}."
            )

    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    label_map = ckpt["label_map"]
    inv = {v: k for k, v in label_map.items()}
    labels = [inv[i] for i in range(len(inv))]

    model = _build_model(config)
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return SignLanguageRuntime(model=model, labels=labels, config=config, device=device)


# ---------------------------------------------------------------------------
# MediaPipe landmark extraction
# ---------------------------------------------------------------------------

_HOLISTIC = None

def _get_holistic():
    """Lazily initialize a single MediaPipe Holistic instance."""
    global _HOLISTIC
    if _HOLISTIC is None:
        import mediapipe as mp
        _HOLISTIC = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    return _HOLISTIC


def _flatten_lm(landmarks, n: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros(n * 3, dtype=np.float32)
    return np.array(
        [[p.x, p.y, p.z] for p in landmarks.landmark],
        dtype=np.float32,
    ).flatten()


def _landmarks_from_result(result) -> np.ndarray:
    lh = _flatten_lm(result.left_hand_landmarks, 21)
    rh = _flatten_lm(result.right_hand_landmarks, 21)
    pose = _flatten_lm(result.pose_landmarks, 33)
    return np.concatenate([lh, rh, pose]).astype(np.float32)


def process_frame_for_landmarks(frame_bgr_or_rgb: np.ndarray):
    """Run MediaPipe once and return both landmark features and raw tracking."""
    # Gradio streams RGB numpy arrays, OpenCV capture returns BGR.
    # MediaPipe expects RGB. We assume RGB (the common streaming case);
    # if the caller has BGR they should convert first.
    if frame_bgr_or_rgb.ndim != 3:
        return np.zeros(FEAT_DIM, dtype=np.float32), None

    holistic = _get_holistic()
    result = holistic.process(frame_bgr_or_rgb)
    return _landmarks_from_result(result), result


def extract_landmarks_from_frame(frame_bgr_or_rgb: np.ndarray) -> np.ndarray:
    """Extract a 225-d landmark vector from a single BGR or RGB frame."""
    landmarks, _ = process_frame_for_landmarks(frame_bgr_or_rgb)
    return landmarks


def _normalized_points(landmarks, width: int, height: int) -> list[tuple[int, int]]:
    if landmarks is None:
        return []
    points = []
    for point in landmarks.landmark:
        x = int(np.clip(point.x, 0.0, 1.0) * width)
        y = int(np.clip(point.y, 0.0, 1.0) * height)
        points.append((x, y))
    return points


def _draw_landmark_group(
    frame: np.ndarray,
    landmarks,
    label: str,
    color: tuple[int, int, int],
) -> bool:
    """Draw landmark dots and a bounding box for one MediaPipe group."""
    import cv2

    height, width = frame.shape[:2]
    points = _normalized_points(landmarks, width, height)
    if not points:
        return False

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 12
    x1 = max(min(xs) - pad, 0)
    y1 = max(min(ys) - pad, 0)
    x2 = min(max(xs) + pad, width - 1)
    y2 = min(max(ys) + pad, height - 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    for x, y in points:
        cv2.circle(frame, (x, y), 3, color, -1)
    return True


def draw_tracking_overlay(
    frame_rgb: np.ndarray,
    tracking_result,
    label: str | None = None,
    confidence: float | None = None,
    top5: list[tuple[str, float]] | None = None,
    buffer_size: int = 0,
) -> np.ndarray:
    """Draw MediaPipe tracking boxes and current model state on an RGB frame."""
    import cv2

    if frame_rgb.ndim != 3:
        return frame_rgb

    annotated = np.ascontiguousarray(frame_rgb.copy())
    if tracking_result is None:
        return annotated

    left_seen = _draw_landmark_group(
        annotated,
        tracking_result.left_hand_landmarks,
        "left hand",
        (0, 210, 255),
    )
    right_seen = _draw_landmark_group(
        annotated,
        tracking_result.right_hand_landmarks,
        "right hand",
        (80, 255, 120),
    )
    pose_seen = _draw_landmark_group(
        annotated,
        tracking_result.pose_landmarks,
        "pose",
        (255, 170, 40),
    )

    status = []
    if left_seen:
        status.append("L hand")
    if right_seen:
        status.append("R hand")
    if pose_seen:
        status.append("pose")
    status_text = "Tracking: " + (", ".join(status) if status else "no landmarks")

    panel_h = 88 if top5 else 58
    cv2.rectangle(annotated, (10, 10), (560, panel_h), (12, 12, 12), -1)
    cv2.rectangle(annotated, (10, 10), (560, panel_h), (255, 255, 255), 1)
    cv2.putText(
        annotated,
        status_text,
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if label:
        label_text = label.replace("_", " ").title()
        if confidence is not None:
            label_text = f"Candidate: {label_text} ({confidence:.0%})"
        else:
            label_text = f"Candidate: {label_text}"
    else:
        label_text = f"Collecting frames: {buffer_size}/{SEQ_LEN_DEFAULT}"
    cv2.putText(
        annotated,
        label_text,
        (22, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 220, 120),
        2,
        cv2.LINE_AA,
    )

    if top5:
        top3 = ", ".join(
            f"{name.replace('_', ' ')} {score:.0%}"
            for name, score in top5[:3]
        )
        cv2.putText(
            annotated,
            f"Top: {top3}",
            (22, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 230, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def extract_landmarks_from_video(
    video_path: str | Path,
    seq_len: int = SEQ_LEN_DEFAULT,
) -> np.ndarray | None:
    """Read a video file, uniformly sample `seq_len` frames, return (seq_len, 225)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None

    idx = np.linspace(0, len(frames) - 1, seq_len).astype(int)
    sampled = [frames[i] for i in idx]

    holistic = _get_holistic()
    seq = np.zeros((seq_len, FEAT_DIM), dtype=np.float32)
    for t, frame_bgr in enumerate(sampled):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        lh = _flatten_lm(result.left_hand_landmarks, 21)
        rh = _flatten_lm(result.right_hand_landmarks, 21)
        pose = _flatten_lm(result.pose_landmarks, 33)
        seq[t] = np.concatenate([lh, rh, pose])
    return seq
