from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import app


def test_reset_live_returns_state_and_status() -> None:
    state, sign_status, sentence_status = app.reset_live()

    assert "buffer" in state
    assert "Caption" in sign_status
    assert "Sentence Watch" in sentence_status


def test_classify_sentence_clip_uses_retrieval_mode(monkeypatch) -> None:
    class FakeRuntime:
        frame_count = 16
        image_size = 112
        model_path = "models/cnn-3d-sentence-top25.keras"
        inference_mode = "embedding_knn"
        embedding_model = object()
        embedding_index = SimpleNamespace(
            index_path="models/cnn-3d-sentence-top25.index.npz",
            support_split="all",
        )

        def predict_clip_retrieval(self, clip_volume):
            return (
                "Thank you.",
                0.973,
                [("Thank you.", 0.973), ("Good.", 0.812)],
                {"neighbor_clip_ids": ["clip_a", "clip_b"]},
            )

    monkeypatch.setattr(app, "_require_sentence_runtime", lambda: FakeRuntime(), raising=False)
    monkeypatch.setattr(
        app,
        "extract_clip_volume",
        lambda *args, **kwargs: (
            np.zeros((16, 112, 112, 3), dtype=np.float32),
            {"sampled_frames": 16, "source_frames": 16},
        ),
        raising=False,
    )

    markdown, details = app.classify_sentence_clip("demo.mp4")

    assert "Best Sentence Match" in markdown
    assert "Thank you." in markdown
    assert "Inference mode: embedding_knn" in markdown
    assert "Support set: all" in markdown
    assert "Top Sentence Matches" not in markdown
    assert details["accepted_label"] == "Thank you."
    assert details["sentence_inference_mode"] == "embedding_knn"
    assert "top5" not in details
