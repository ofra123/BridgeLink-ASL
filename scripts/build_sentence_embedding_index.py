"""Build a nearest-neighbor sentence index from 3D CNN clip embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 3D CNN sentence embedding index.")
    parser.add_argument(
        "--manifest",
        default="data/processed/how2sign_sentences_top25.frames.jsonl",
        help="Clip manifest JSONL with sampled_frames.",
    )
    parser.add_argument(
        "--model",
        default="models/cnn-3d-sentence-top25.keras",
        help="Path to the trained Keras sentence CNN.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the .index.npz file. Defaults next to the model.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "val", "test"),
        help="Which manifest split to index. Use 'all' for the full support set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Voting neighborhood size for retrieval.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=10,
        help="How many nearest clips to scan when ranking unique labels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from bridgelink_asl.clip_dataset import load_clip_dataset
    from bridgelink_asl.cnn import select_frame_paths
    from bridgelink_asl.sentence_inference import load_sentence_runtime
    import tensorflow as tf

    manifest_path = Path(args.manifest).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path.with_suffix(".index.npz")
    )

    runtime = load_sentence_runtime(local_model_path=model_path)
    if runtime.embedding_model is None:
        raise RuntimeError("The sentence model does not expose a clip_embedding layer.")

    records = load_clip_dataset(manifest_path)
    if args.split != "all":
        records = [record for record in records if record.split == args.split]
    if not records:
        raise RuntimeError(f"No records found for split={args.split!r} in {manifest_path}.")

    clip_ids = [record.clip_id for record in records]
    labels = [record.label for record in records]
    selected_paths = [
        [str(path) for path in select_frame_paths(record.sampled_frames, runtime.frame_count)]
        for record in records
    ]

    def _load_clip(paths):
        images = tf.map_fn(
            lambda path: _load_image(tf, path, runtime.image_size, runtime.channels),
            paths,
            fn_output_signature=tf.float32,
        )
        return images

    dataset = (
        tf.data.Dataset.from_tensor_slices(selected_paths)
        .map(_load_clip, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    embeddings = np.asarray(runtime.embedding_model.predict(dataset, verbose=0), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = embeddings / norms

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        normalized_embeddings=normalized.astype(np.float32),
        labels=np.asarray(labels, dtype="<U128"),
        clip_ids=np.asarray(clip_ids, dtype="<U128"),
        source_manifest=np.asarray(str(manifest_path), dtype="<U512"),
        support_split=np.asarray(args.split, dtype="<U32"),
        top_k=np.asarray(args.top_k, dtype=np.int32),
        candidate_pool=np.asarray(args.candidate_pool, dtype=np.int32),
    )
    print(f"Wrote sentence embedding index to {output_path}")
    print(f"Indexed clips: {len(records)}")


def _load_image(tf, path, image_size: int, channels: int):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=channels, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size))
    return tf.cast(image, tf.float32)


if __name__ == "__main__":
    main()
