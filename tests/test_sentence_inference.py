from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from bridgelink_asl import sentence_inference
from bridgelink_asl.sentence_inference import (
    SentenceClipRuntime,
    SentenceEmbeddingIndex,
    extract_clip_volume,
    load_sentence_runtime,
)


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


def test_sentence_embedding_index_votes_for_best_label() -> None:
    index = SentenceEmbeddingIndex(
        normalized_embeddings=np.array(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        labels=["Thank you.", "Thank you.", "Good."],
        clip_ids=["thankyou_a", "thankyou_b", "good_a"],
        top_k=3,
        candidate_pool=3,
    )

    label, similarity, top5, metadata = index.classify(np.array([1.0, 0.0], dtype=np.float32))

    assert label == "Thank you."
    assert similarity == pytest.approx(1.0)
    assert top5[0][0] == "Thank you."
    assert metadata["neighbor_labels"][0] == "Thank you."


def test_load_sentence_runtime_loads_embedding_index(tmp_path: Path) -> None:
    model_path = tmp_path / "sentence.keras"
    labels_path = tmp_path / "sentence.labels.json"
    index_path = tmp_path / "sentence.index.npz"
    model_path.write_bytes(b"keras")
    labels_path.write_text(json.dumps({"labels": ["Hello!", "Thank you."]}), encoding="utf-8")
    np.savez_compressed(
        index_path,
        normalized_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=np.array(["Hello!", "Thank you."], dtype="<U32"),
        clip_ids=np.array(["clip_a", "clip_b"], dtype="<U32"),
        source_manifest=np.array(str(tmp_path / "manifest.jsonl"), dtype="<U256"),
        top_k=np.array(3, dtype=np.int32),
        candidate_pool=np.array(10, dtype=np.int32),
    )

    fake_model = mock.Mock()
    fake_model.input_shape = (None, 16, 112, 112, 3)
    fake_model.input = object()
    fake_model.get_layer.return_value = SimpleNamespace(output=object())
    fake_embedding_model = mock.Mock()

    fake_tf = mock.Mock()
    fake_tf.keras.models.load_model.return_value = fake_model
    fake_tf.keras.Model.return_value = fake_embedding_model

    with mock.patch.object(sentence_inference, "_require_tensorflow", return_value=fake_tf):
        runtime = load_sentence_runtime(
            local_model_path=model_path,
            local_labels_path=labels_path,
        )

    assert runtime.embedding_model is fake_embedding_model
    assert runtime.embedding_index is not None
    assert runtime.embedding_index.index_path == str(index_path)
    assert runtime.inference_mode == "embedding_knn"


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
