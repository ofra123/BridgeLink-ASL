"""Sentence-level 3D CNN baseline for closed-vocabulary video classification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .clip_dataset import ClipDatasetRecord, load_clip_dataset, summarize_clip_splits, validate_clip_dataset


@dataclass(frozen=True)
class CnnModelConfig:
    """Configuration for the sentence-level 3D CNN baseline."""

    frame_count: int = 16
    image_size: int = 112
    channels: int = 3
    batch_size: int = 2
    epochs: int = 12
    learning_rate: float = 0.001
    dropout: float = 0.3
    conv_filters: tuple[int, ...] = (16, 32, 64)
    model_path: Path = field(default_factory=lambda: Path("models/cnn-3d-sentence.keras"))
    manifest_path: Path = field(default_factory=lambda: Path("data/processed/how2sign_sentences_top12.jsonl"))

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (self.frame_count, self.image_size, self.image_size, self.channels)


@dataclass(frozen=True)
class CnnTrainingPlan:
    """Dry-run summary for the sentence-level CNN branch."""

    labels: tuple[str, ...]
    split_distribution: dict[str, int]
    input_shape: tuple[int, int, int, int]
    train_records: int
    val_records: int
    test_records: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "split_distribution": self.split_distribution,
            "input_shape": list(self.input_shape),
            "train_records": self.train_records,
            "val_records": self.val_records,
            "test_records": self.test_records,
        }


@dataclass(frozen=True)
class CnnTrainingSummary:
    """Result from a real 3D CNN training run."""

    manifest_path: str
    output_path: str
    labels: tuple[str, ...]
    epochs: int
    split_distribution: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "output_path": self.output_path,
            "labels": list(self.labels),
            "epochs": self.epochs,
            "split_distribution": self.split_distribution,
        }


def describe_cnn_baseline(config: CnnModelConfig, num_classes: int) -> dict[str, Any]:
    """Describe the 3D CNN architecture without importing TensorFlow."""

    return {
        "model_family": "sentence-3d-cnn",
        "comparison_role": "Sentence-level RGB video baseline against VLM sentence interpretation",
        "input_shape": list(config.input_shape),
        "num_classes": num_classes,
        "temporal_strategy": "Conv3D over sampled RGB clip volumes",
        "conv_filters": list(config.conv_filters),
        "dropout": config.dropout,
    }


def build_cnn_training_plan(records: Iterable[ClipDatasetRecord], config: CnnModelConfig) -> CnnTrainingPlan:
    """Build a dry-run training plan from sentence clip metadata."""

    materialized = list(records)
    labels = tuple(sorted({record.label for record in materialized}))
    split_distribution = summarize_clip_splits(materialized)
    return CnnTrainingPlan(
        labels=labels,
        split_distribution=split_distribution,
        input_shape=config.input_shape,
        train_records=split_distribution.get("train", 0),
        val_records=split_distribution.get("val", 0),
        test_records=split_distribution.get("test", 0),
    )


def select_frame_paths(frame_paths: Iterable[Path], frame_count: int) -> tuple[Path, ...]:
    """Uniformly sample or pad frame paths to a fixed clip length."""

    paths = tuple(frame_paths)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if not paths:
        raise ValueError("At least one frame path is required.")
    if len(paths) == frame_count:
        return paths
    if len(paths) < frame_count:
        return paths + (paths[-1],) * (frame_count - len(paths))

    if frame_count == 1:
        return (paths[0],)

    last_index = len(paths) - 1
    indexes = [round(index * last_index / (frame_count - 1)) for index in range(frame_count)]
    return tuple(paths[index] for index in indexes)


def build_clip_cnn_model(num_classes: int, config: CnnModelConfig):
    """Build and compile the TensorFlow/Keras 3D CNN."""

    tf = _require_tensorflow()
    if num_classes <= 1:
        raise ValueError("CNN training requires at least two sentence classes.")

    inputs = tf.keras.Input(shape=config.input_shape, name="clip_frames")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_frames")(inputs)
    for index, filters in enumerate(config.conv_filters):
        x = tf.keras.layers.Conv3D(
            filters,
            kernel_size=(3, 3, 3),
            padding="same",
            activation="relu",
            name=f"conv3d_{index + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn3d_{index + 1}")(x)
        pool = (1, 2, 2) if index == 0 else (2, 2, 2)
        x = tf.keras.layers.MaxPooling3D(pool_size=pool, name=f"pool3d_{index + 1}")(x)

    x = tf.keras.layers.GlobalAveragePooling3D(name="clip_embedding")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="sentence_class")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bridgelink_sentence_3dcnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_clip_cnn_model(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    config: CnnModelConfig | None = None,
) -> CnnTrainingSummary:
    """Train the 3D CNN baseline from sampled frame paths in a clip manifest."""

    active_config = config or CnnModelConfig()
    manifest = Path(manifest_path).expanduser().resolve()
    records = load_clip_dataset(manifest)
    issues = validate_clip_dataset(records, require_sampled_frames=True, require_files=True)
    if issues:
        raise ValueError("; ".join(issues))

    plan = build_cnn_training_plan(records, active_config)
    if plan.train_records == 0:
        raise ValueError("CNN training requires at least one training clip.")
    if len(plan.labels) <= 1:
        raise ValueError("CNN training requires at least two sentence classes.")

    tf = _require_tensorflow()
    label_to_index = {label: index for index, label in enumerate(plan.labels)}
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]

    model = build_clip_cnn_model(num_classes=len(plan.labels), config=active_config)
    train_dataset = _build_tf_dataset(tf, train_records, label_to_index, active_config, shuffle=True)
    val_dataset = _build_tf_dataset(tf, val_records, label_to_index, active_config, shuffle=False) if val_records else None
    model.fit(train_dataset, validation_data=val_dataset, epochs=active_config.epochs)

    save_path = Path(output_path or active_config.model_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    label_path = save_path.with_suffix(".labels.json")
    label_path.write_text(json.dumps({"labels": list(plan.labels)}, indent=2), encoding="utf-8")

    return CnnTrainingSummary(
        manifest_path=str(manifest),
        output_path=str(save_path),
        labels=plan.labels,
        epochs=active_config.epochs,
        split_distribution=plan.split_distribution,
    )


def _build_tf_dataset(
    tf: Any,
    records: Iterable[ClipDatasetRecord],
    label_to_index: dict[str, int],
    config: CnnModelConfig,
    *,
    shuffle: bool,
):
    selected_paths = [
        [str(path) for path in select_frame_paths(record.sampled_frames, config.frame_count)]
        for record in records
    ]
    labels = [label_to_index[record.label] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((selected_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(len(labels), 1), reshuffle_each_iteration=True)

    def load_clip(paths, label):
        images = tf.map_fn(
            lambda path: _load_image(tf, path, config),
            paths,
            fn_output_signature=tf.float32,
        )
        return images, label

    return dataset.map(load_clip).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def _load_image(tf: Any, path: Any, config: CnnModelConfig):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=config.channels, expand_animations=False)
    image = tf.image.resize(image, (config.image_size, config.image_size))
    return tf.cast(image, tf.float32)


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for sentence CNN training. Install it with "
            "`pip install -e .[training]` before running the CNN trainer."
        ) from exc
    return tf
