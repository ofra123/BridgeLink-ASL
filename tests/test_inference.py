"""Smoke tests for the inference module."""

import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure src is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_model_builds():
    """The SignTransformer can be instantiated and produce logits."""
    from bridgelink_asl.inference import _build_model

    config = {
        "num_classes": 10,
        "d_model": 64,
        "nhead": 2,
        "layers": 2,
        "seq_len": 32,
        "feat_dim": 225,
        "dropout": 0.1,
    }
    model = _build_model(config)
    import torch

    x = torch.randn(2, 32, 225)
    logits = model(x)
    assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"


def test_cnn_model_builds():
    """The landmark CNN can be instantiated and produce logits."""
    from bridgelink_asl.inference import _build_model
    import torch

    config = {
        "model_type": "landmark_cnn",
        "num_classes": 10,
        "feat_dim": 225,
        "channels": [32, 64, 64],
        "dropout": 0.1,
    }
    model = _build_model(config)
    x = torch.randn(2, 32, 225)
    logits = model(x)
    assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"


def test_extract_landmarks_from_frame():
    """Landmark extraction returns a 225-d vector from a dummy frame."""
    if importlib.util.find_spec("mediapipe") is None:
        pytest.skip("MediaPipe is not installed in this local test environment")

    from bridgelink_asl.inference import extract_landmarks_from_frame, FEAT_DIM

    # Black frame — MediaPipe probably won't detect hands, but the function
    # should still return a zero-padded vector without crashing.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = extract_landmarks_from_frame(frame)
    assert lm.shape == (FEAT_DIM,), f"Expected ({FEAT_DIM},), got {lm.shape}"
    assert lm.dtype == np.float32


def test_runtime_predict():
    """A dummy runtime can classify a random sequence."""
    from bridgelink_asl.inference import _build_model, SignLanguageRuntime
    import torch

    config = {"num_classes": 5, "d_model": 64, "nhead": 2, "layers": 2,
              "seq_len": 32, "feat_dim": 225, "dropout": 0.1}
    model = _build_model(config)
    model.eval()

    labels = ["a", "b", "c", "d", "e"]
    runtime = SignLanguageRuntime(model=model, labels=labels, config=config, device="cpu")

    seq = np.random.randn(32, 225).astype(np.float32)
    label, confidence, top5 = runtime.predict(seq)

    assert label in labels
    assert 0.0 <= confidence <= 1.0
    assert len(top5) == 5
    assert all(name in labels for name, _ in top5)
