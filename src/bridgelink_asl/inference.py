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

def _build_model(config: dict):
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
    local_path: Path | str = "models/sign_transformer_best.pt",
    hf_repo: Optional[str] = None,
) -> SignLanguageRuntime:
    """Load model weights from a local path, with optional HF Hub fallback."""
    import torch

    local_path = Path(local_path)

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=hf_repo, filename="sign_transformer_best.pt")
            local_path = Path(downloaded)
        except Exception as exc:
            print(f"[bridgelink] HF Hub download failed ({exc}); falling back to local path")

    if not local_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {local_path}. "
            f"Train the model with the Kaggle notebook and place the .pt file there, "
            f"or set HF_MODEL_REPO to a HF model repo containing sign_transformer_best.pt."
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


def extract_landmarks_from_frame(frame_bgr_or_rgb: np.ndarray) -> np.ndarray:
    """Extract a 225-d landmark vector from a single BGR or RGB frame."""
    # Gradio streams RGB numpy arrays, OpenCV capture returns BGR.
    # MediaPipe expects RGB. We assume RGB (the common streaming case);
    # if the caller has BGR they should convert first.
    if frame_bgr_or_rgb.ndim != 3:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    holistic = _get_holistic()
    result = holistic.process(frame_bgr_or_rgb)
    lh = _flatten_lm(result.left_hand_landmarks, 21)
    rh = _flatten_lm(result.right_hand_landmarks, 21)
    pose = _flatten_lm(result.pose_landmarks, 33)
    return np.concatenate([lh, rh, pose]).astype(np.float32)


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
