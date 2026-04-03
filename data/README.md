# Data Layout

The starter repo separates data by maturity:

- `raw/`: captured clips, images, or exported landmark dumps
- `processed/`: cleaned landmark vectors ready for training
- `interim/`: temporary conversion outputs

`processed/sample_landmarks.jsonl` is a small bootstrap dataset that lets the team exercise the training and evaluation commands without waiting on collection.

Each JSONL record should follow this shape:

```json
{"label": "HELLO", "split": "train", "landmarks": [0.12, 0.18, 0.81, 0.75, 0.22, 0.68]}
```

For the real class dataset, keep the following conventions:

- labels should match the V1 vocabulary in `docs/v1-scope.md`
- splits should be one of `train`, `val`, or `test`
- every sample should have the same landmark feature length
- raw video should remain outside Git unless it is tiny and explicitly approved
