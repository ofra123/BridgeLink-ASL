from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from bridgelink_asl import sentence_inference
from bridgelink_asl.sentence_inference import SentenceClipRuntime, extract_clip_volume, load_sentence_runtime


def test_load_sentence_runtime_reads_labels_and_shape(tmp_path: Path) -> None:
    model_path = tmp_path / "sentence.keras"
    labels_path = tmp_path / "sentence.labels.json"
    model_path.write_bytes(b"keras")
    labels_path.write_text(json.dumps({"labels": ["Hello!", "Thank you."]}), encoding="utf-8")

    fake_model = mock.Mock()
    fake_model.input_shape = (None, 16, 112, 112, 3)

    fake_tf = mock.Mock()
    fake_tf.keras.models.load_model.return_value = fake_model

    with mock.patch.object(sentence_inference, "_require_tensorflow", return_value=fake_tf):
        runtime = load_sentence_runtime(
            local_model_path=model_path,
            local_labels_path=labels_path,
        )

    assert runtime.labels == ["Hello!", "Thank you."]
    assert runtime.frame_count == 16
    assert runtime.image_size == 112
    assert runtime.channels == 3
    fake_tf.keras.models.load_model.assert_called_once_with(model_path)


def test_sentence_runtime_predict_clip_returns_topk() -> None:
    fake_model = mock.Mock()
    fake_model.predict.return_value = np.array([[0.1, 0.75, 0.15]], dtype=np.float32)
    runtime = SentenceClipRuntime(
        model=fake_model,
        labels=["Hello!", "Thank you.", "Yes."],
        frame_count=16,
        image_size=112,
        channels=3,
        model_path="mock.keras",
    )

    clip = np.zeros((16, 112, 112, 3), dtype=np.float32)
    label, confidence, top5 = runtime.predict_clip(clip)

    assert label == "Thank you."
    assert confidence == pytest.approx(0.75)
    assert top5[0][0] == "Thank you."
    fake_model.predict.assert_called_once()


def test_extract_clip_volume_samples_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")

    video_path = tmp_path / "demo.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (32, 32),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")

    for index in range(5):
        frame = np.full((32, 32, 3), 40 * index, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    clip, metadata = extract_clip_volume(video_path, frame_count=8, image_size=16)

    assert clip is not None
    assert clip.shape == (8, 16, 16, 3)
    assert metadata["source_frames"] == 5
    assert metadata["sampled_frames"] == 8
