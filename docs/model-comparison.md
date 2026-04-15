# CNN vs VLM Comparison Plan

## Goal

The graded comparison is CNN baseline versus VLM sentence interpreter over the same sentence clip set.

## Models

- CNN baseline: sampled clip frames -> sentence/gloss class. This is Omar's model.
- VLM interpreter: sampled clip frames plus optional token trace -> natural English sentence. This is Frank's model.

The existing landmark/centroid word baseline stays useful as a fallback and optional VLM grounding trace, but it is not the main comparison model.

## Shared Input Contract

Both models consume records from the clip manifest:

```json
{
  "clip_id": "team_hello_want_drink_001",
  "split": "test",
  "source": "team-recorded",
  "gloss": ["HELLO", "WANT", "DRINK"],
  "english": "Hello, I want a drink.",
  "video_path": "data/raw/team_hello_want_drink_001.mp4",
  "sampled_frames": [
    "data/interim/frames/team_hello_want_drink_001/frame_0001.jpg"
  ],
  "landmarks_path": null,
  "notes": "Controlled lighting, front-facing signer."
}
```

Large videos and frame dumps stay out of Git. Commit only manifests, scripts, and small metadata samples.

## Metrics

- CNN accuracy: exact gloss-sequence class match.
- VLM accuracy: exact or rubric-scored English sentence match.
- Shared metrics: latency per clip, failure count, missing-data count, and qualitative mistakes.

## Current Implementation

- `clip_dataset.py` loads and validates clip manifests.
- `cnn.py` defines the optional TensorFlow/Keras sampled-frame CNN.
- `train_cnn_model --dry-run` validates the manifest and prints the CNN plan without requiring TensorFlow.
- `train_cnn_model` without `--dry-run` trains the CNN when sampled frame paths and `.[training]` dependencies are available.
