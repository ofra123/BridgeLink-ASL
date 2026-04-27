"""Inference helpers for the How2Sign sentence-level 3D CNN."""

from __future__ import annotations

from collections import defaultdict
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SentenceEmbeddingIndex:
    """Nearest-neighbor sentence index over 3D CNN clip embeddings."""

    normalized_embeddings: np.ndarray
    labels: list[str]
    clip_ids: list[str]
    top_k: int = 3
    candidate_pool: int = 10
    source_manifest: str | None = None
    index_path: str | None = None

    def classify(
        self,
        embedding: np.ndarray,
    ) -> tuple[str, float, list[tuple[str, float]], dict[str, Any]]:
        """Return the best sentence match from the indexed training embeddings."""

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm == 0.0:
            raise ValueError("Sentence embedding has zero norm.")
        normalized = vector / vector_norm
        similarities = self.normalized_embeddings @ normalized

        candidate_pool = min(self.candidate_pool, len(similarities))
        nearest = np.argsort(similarities)[::-1][:candidate_pool]
        vote_nearest = nearest[: min(self.top_k, len(nearest))]

        vote_scores: dict[str, float] = defaultdict(float)
        for index in vote_nearest:
            vote_scores[self.labels[int(index)]] += float(similarities[int(index)])
        label, _ = max(vote_scores.items(), key=lambda item: item[1])

        best_label_similarity = max(
            float(similarities[int(index)])
            for index in nearest
            if self.labels[int(index)] == label
        )

        ranked_unique: list[tuple[str, float]] = [(label, best_label_similarity)]
        seen_labels: set[str] = {label}
        for index in nearest:
            candidate_label = self.labels[int(index)]
            if candidate_label in seen_labels:
                continue
            seen_labels.add(candidate_label)
            ranked_unique.append((candidate_label, float(similarities[int(index)])))
            if len(ranked_unique) == 5:
                break

        metadata = {
            "neighbor_clip_ids": [self.clip_ids[int(index)] for index in vote_nearest],
            "neighbor_labels": [self.labels[int(index)] for index in vote_nearest],
            "neighbor_similarities": [float(similarities[int(index)]) for index in vote_nearest],
            "source_manifest": self.source_manifest,
            "index_path": self.index_path,
            "top_k": self.top_k,
            "candidate_pool": self.candidate_pool,
        }
        return label, best_label_similarity, ranked_unique, metadata


@dataclass
class SentenceClipRuntime:
    """Runtime wrapper for a Keras sentence clip classifier."""

    model: Any
    labels: list[str]
    frame_count: int
    image_size: int
    channels: int
    model_path: str
    embedding_model: Any | None = None
    embedding_index: SentenceEmbeddingIndex | None = None

    @property
    def inference_mode(self) -> str:
        if self.embedding_model is not None and self.embedding_index is not None:
            return "embedding_knn"
        return "softmax"

    def predict_clip_softmax(self, clip_volume: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """Run the original softmax sentence classifier head."""

        batch = np.expand_dims(clip_volume.astype(np.float32), axis=0)
        probabilities = np.asarray(self.model.predict(batch, verbose=0))[0]
        top5_idx = probabilities.argsort()[::-1][:5]
        top5 = [(self.labels[int(i)], float(probabilities[int(i)])) for i in top5_idx]
        best_idx = int(probabilities.argmax())
        return self.labels[best_idx], float(probabilities[best_idx]), top5

    def embed_clip(self, clip_volume: np.ndarray) -> np.ndarray:
        """Project one clip volume into the penultimate 3D CNN embedding space."""

        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not available for sentence retrieval.")
        batch = np.expand_dims(clip_volume.astype(np.float32), axis=0)
        embedding = np.asarray(self.embedding_model.predict(batch, verbose=0))[0]
        return embedding

    def predict_clip_retrieval(
        self,
        clip_volume: np.ndarray,
    ) -> tuple[str, float, list[tuple[str, float]], dict[str, Any]]:
        """Classify a clip by nearest-neighbor matching in embedding space."""

        if self.embedding_index is None:
            raise RuntimeError("Sentence embedding index is not available.")
        embedding = self.embed_clip(clip_volume)
        return self.embedding_index.classify(embedding)

    def predict_clip(self, clip_volume: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """Run inference on one clip volume with shape (T, H, W, C)."""

        if self.embedding_model is not None and self.embedding_index is not None:
            label, score, top5, _ = self.predict_clip_retrieval(clip_volume)
            return label, score, top5
        return self.predict_clip_softmax(clip_volume)


def load_sentence_runtime(
    local_model_path: str | Path = "models/cnn-3d-sentence-top25.keras",
    *,
    local_labels_path: str | Path | None = None,
    hf_repo: str | None = None,
    hf_model_filename: str | None = None,
    hf_labels_filename: str | None = None,
) -> SentenceClipRuntime:
    """Load a trained How2Sign sentence classifier from disk or the HF Hub."""

    tf = _require_tensorflow()
    model_path = Path(local_model_path)
    model_filename = hf_model_filename or model_path.name
    labels_path = Path(local_labels_path) if local_labels_path else model_path.with_suffix(".labels.json")
    labels_filename = hf_labels_filename or labels_path.name
    index_path = model_path.with_suffix(".index.npz")
    index_filename = index_path.name

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download

            model_path = Path(hf_hub_download(repo_id=hf_repo, filename=model_filename))
            labels_path = Path(hf_hub_download(repo_id=hf_repo, filename=labels_filename))
            try:
                index_path = Path(hf_hub_download(repo_id=hf_repo, filename=index_filename))
            except Exception:
                index_path = model_path.with_suffix(".index.npz")
        except Exception as exc:
            print(f"[bridgelink] HF Hub download failed for sentence model ({exc}); falling back to local files")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Sentence CNN weights not found at {model_path}. "
            f"Upload {model_filename} to the repo or set HF_SENTENCE_MODEL_REPO."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Sentence CNN labels not found at {labels_path}. "
            f"Upload {labels_filename} to the repo or set HF_SENTENCE_MODEL_REPO."
        )

    model = tf.keras.models.load_model(model_path)
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = list(payload["labels"])
    input_shape = tuple(model.input_shape)
    if len(input_shape) != 5:
        raise ValueError(f"Unexpected sentence CNN input shape: {input_shape}")

    _, frame_count, image_height, image_width, channels = input_shape
    if image_height != image_width:
        raise ValueError(f"Sentence CNN expects square frames, got {input_shape}")

    embedding_model = None
    embedding_index = None
    try:
        embedding_model = tf.keras.Model(model.input, model.get_layer("clip_embedding").output)
    except Exception:
        embedding_model = None
    if index_path.exists():
        embedding_index = _load_embedding_index(index_path)

    return SentenceClipRuntime(
        model=model,
        labels=labels,
        frame_count=int(frame_count),
        image_size=int(image_height),
        channels=int(channels),
        model_path=str(model_path),
        embedding_model=embedding_model,
        embedding_index=embedding_index,
    )


def extract_clip_volume(
    video_path: str | Path,
    *,
    frame_count: int,
    image_size: int,
) -> tuple[np.ndarray | None, dict[str, float | int | None]]:
    """Decode a video file and convert it into a fixed-size RGB clip volume."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None, {"source_frames": 0, "sampled_frames": 0, "fps": None, "duration_seconds": None}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        return None, {"source_frames": 0, "sampled_frames": 0, "fps": fps or None, "duration_seconds": None}

    indexes = _uniform_indexes(len(frames), frame_count)
    clip_frames: list[np.ndarray] = []
    for index in indexes:
        rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        clip_frames.append(resized.astype(np.float32))

    duration_seconds = None
    if fps > 0:
        duration_seconds = round((reported_frame_count or len(frames)) / fps, 2)

    metadata: dict[str, float | int | None] = {
        "source_frames": len(frames),
        "sampled_frames": len(clip_frames),
        "fps": round(fps, 2) if fps > 0 else None,
        "duration_seconds": duration_seconds,
    }
    return np.stack(clip_frames, axis=0), metadata


def _uniform_indexes(total_frames: int, frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    return np.rint(np.linspace(0, total_frames - 1, frame_count)).astype(int)


def _load_embedding_index(index_path: str | Path) -> SentenceEmbeddingIndex:
    payload = np.load(Path(index_path), allow_pickle=False)
    source_manifest = None
    if "source_manifest" in payload:
        source_manifest = str(payload["source_manifest"].item())
    top_k = int(payload["top_k"].item()) if "top_k" in payload else 3
    candidate_pool = int(payload["candidate_pool"].item()) if "candidate_pool" in payload else 10
    return SentenceEmbeddingIndex(
        normalized_embeddings=np.asarray(payload["normalized_embeddings"], dtype=np.float32),
        labels=[str(item) for item in payload["labels"].tolist()],
        clip_ids=[str(item) for item in payload["clip_ids"].tolist()],
        top_k=top_k,
        candidate_pool=candidate_pool,
        source_manifest=source_manifest,
        index_path=str(Path(index_path)),
    )


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for the How2Sign sentence classifier. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc
    return tf
