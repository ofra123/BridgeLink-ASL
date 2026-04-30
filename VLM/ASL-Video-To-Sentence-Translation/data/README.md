# Data Directory

This directory is intentionally empty in git. You do not have to store the raw
dataset here, but you can if you want.

The training/evaluation scripts read small prepared JSONL files from here:

- `asl_citizen_train.jsonl`
- `asl_citizen_val.jsonl`
- `how2sign_train.jsonl`
- `how2sign_val.jsonl`

Each JSONL row stores a path to a local video file and the target text. Paths
may be absolute or relative to the repository root, but no script hard-codes
local machine paths.

## Recommended Local Layout

If you want a simple layout, put downloaded datasets under `data/raw/`:

```text
data/
  raw/
    asl_citizen/
      metadata.csv
      videos/
        example_001.mp4
        example_002.mp4
```

Then run:

```bash
python scripts/prepare_asl_citizen.py \
  --metadata data/raw/asl_citizen/metadata.csv \
  --video_root data/raw/asl_citizen/videos \
  --out_train data/asl_citizen_train.jsonl \
  --out_val data/asl_citizen_val.jsonl
```

If your dataset is somewhere else, leave it there and pass those paths instead.
The preparation script does not copy videos. It only creates JSONL rows pointing
to the video files.

## What Metadata Needs

For ASL Citizen, the metadata file can be CSV, JSON, or JSONL. It needs:

- a video column: `video_path`, `path`, `file`, `filename`, or `video`
- a label column: `label`, `gloss`, `sign`, `sign_label`, or `text`
- optionally a split column: `split` or `subset`

If there is no split column, the script makes a train/validation split using
`--val_fraction`.
