from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import app


def test_init_live_state_includes_sentence_buffer(monkeypatch) -> None:
    monkeypatch.setattr(app, "SENTENCE_RUNTIME", SimpleNamespace(frame_count=16), raising=False)

    state = app.init_live_state()

    assert "sentence_buffer" in state
    assert state["sentence_buffer"].maxlen == 16
    assert state["sentence_caption"] == ""


def test_sentence_status_markdown_uses_confidence_gate(monkeypatch) -> None:
    monkeypatch.setattr(app, "SENTENCE_RUNTIME", SimpleNamespace(frame_count=16), raising=False)
    state = app.init_live_state()
    state["sentence_caption"] = "Thank you."
    state["sentence_candidate_label"] = "Thank you."
    state["sentence_candidate_confidence"] = 0.81

    status = app._sentence_status_markdown(state)

    assert "Sentence Watch" in status
    assert "Thank you." in status
    assert "Closed vocabulary" in status


def test_update_sentence_live_state_requires_confidence(monkeypatch) -> None:
    class FakeRuntime:
        frame_count = 1
        image_size = 8

        def predict_clip(self, clip_volume):
            assert clip_volume.shape == (1, 8, 8, 3)
            return "Hello!", 0.42, [("Hello!", 0.42)]

    monkeypatch.setattr(app, "SENTENCE_RUNTIME", FakeRuntime(), raising=False)
    monkeypatch.setattr(app, "SENTENCE_MIN_CONFIDENCE", 0.55, raising=False)
    monkeypatch.setattr(app, "SENTENCE_STREAM_STRIDE", 1, raising=False)

    state = app.init_live_state()
    state["frame_idx"] = 1
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    app._update_sentence_live_state(frame, state)

    assert state["sentence_candidate_label"] is None
    assert state["sentence_caption"] == ""


def test_update_sentence_live_state_emits_after_stability(monkeypatch) -> None:
    class FakeRuntime:
        frame_count = 1
        image_size = 8

        def predict_clip(self, clip_volume):
            return "Thank you.", 0.91, [("Thank you.", 0.91)]

    monkeypatch.setattr(app, "SENTENCE_RUNTIME", FakeRuntime(), raising=False)
    monkeypatch.setattr(app, "SENTENCE_MIN_CONFIDENCE", 0.55, raising=False)
    monkeypatch.setattr(app, "SENTENCE_STREAM_STRIDE", 1, raising=False)
    monkeypatch.setattr(app, "SENTENCE_STABILITY_K", 1, raising=False)

    state = app.init_live_state()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    state["frame_idx"] = 1
    app._update_sentence_live_state(frame, state)

    assert state["sentence_candidate_label"] == "Thank you."
    assert state["sentence_caption"] == "Thank you."


def test_reset_live_returns_both_status_panels(monkeypatch) -> None:
    monkeypatch.setattr(app, "SENTENCE_RUNTIME", SimpleNamespace(frame_count=16), raising=False)

    state, sign_status, sentence_status = app.reset_live()

    assert "buffer" in state
    assert "Caption" in sign_status
    assert "Sentence Watch" in sentence_status
