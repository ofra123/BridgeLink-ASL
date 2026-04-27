# How2Sign Sentence Training

This track is the sentence-level extension for BridgeLink ASL. It uses
**How2Sign frontal RGB sentence clips** and trains a **closed-vocabulary 3D
CNN** over sampled video frames.

## What this branch does

This is **not** open-ended ASL-to-English translation. The realistic first step
is a **sentence classifier** over a repeated subset of How2Sign sentences.

Pipeline:

```text
How2Sign RGB clip
-> sample 16 frames
-> resize to 112 x 112
-> 3D CNN over the clip volume
-> predicted sentence class
```

We use the manually re-aligned translation CSVs to find repeated sentences,
then keep a manageable closed vocabulary, such as the top 12 repeated
sentences with at least 8 examples each.

## Expected raw data layout

Place the downloaded files here:

```text
data/raw/how2sign/
  clips/
    raw_videos/
      *.mp4
  translations/
    how2sign_realigned_train.csv
    how2sign_realigned_val.csv
    how2sign_realigned_test.csv
```

## 1. Build the filtered sentence manifest

Run:

```powershell
python scripts\create_how2sign_sentence_manifest.py `
  --translation-dir data\raw\how2sign\translations `
  --clip-dir data\raw\how2sign\clips\raw_videos `
  --output data\processed\how2sign_sentences_top12.jsonl `
  --min-count 8 `
  --max-classes 12 `
  --max-samples-per-class 20
```

This creates a JSONL manifest of repeated sentences only.

Important:

- The default `top12` setup is intentionally small and usually lands around a
  few hundred clips, not the full raw How2Sign download.
- Training time is driven by the filtered manifest, not the size of the
  original `raw_videos` folder.
- If you want a larger closed-vocabulary run, widen the manifest first, for
  example:

```powershell
python scripts\create_how2sign_sentence_manifest.py `
  --translation-dir data\raw\how2sign\translations `
  --clip-dir data\raw\how2sign\clips\raw_videos `
  --output data\processed\how2sign_sentences_top25.jsonl `
  --min-count 8 `
  --max-classes 25 `
  --max-samples-per-class 60
```

## 2. Normalize duplicate sentence labels

Before frame extraction, we can optionally merge obvious duplicate /
near-duplicate sentence labels. This keeps the same clips but makes the
closed-vocabulary task less noisy.

Current conservative merges:

- `Hi!` + `Hi.` -> `Hi.`
- `O.k.` + `Okay.` + `Okay?` -> `Okay.`
- `Alright.` + `All right.` -> `All right.`

Run:

```powershell
python scripts\normalize_how2sign_sentence_manifest.py `
  --input data\processed\how2sign_sentences_top25.jsonl `
  --output data\processed\how2sign_sentences_top25.normalized.jsonl `
  --summary-output results\how2sign_top25_normalization_summary.json
```

This preserves the same clip count while reducing duplicate classes. In the
current `top25` experiment, it reduced the label space from `25` classes to
`21` classes while keeping all `479` clips.

## 3. Extract sampled frames from each clip

Run:

```powershell
python scripts\prepare_clip_dataset.py `
  --manifest data\processed\how2sign_sentences_top12.jsonl `
  --extract-frames `
  --frame-count 16 `
  --frame-root data\interim\frames\how2sign_top12 `
  --output-manifest data\processed\how2sign_sentences_top12.frames.jsonl `
  --summary-output results\how2sign_top12_dataset_summary.json
```

This updates the manifest with sampled frame paths and writes a dataset summary.

If you are using the normalized `top25` path, run:

```powershell
python scripts\prepare_clip_dataset.py `
  --manifest data\processed\how2sign_sentences_top25.normalized.jsonl `
  --extract-frames `
  --frame-count 16 `
  --frame-root data\interim\frames\how2sign_top25 `
  --output-manifest data\processed\how2sign_sentences_top25.normalized.frames.jsonl `
  --summary-output results\how2sign_top25_dataset_summary.json
```

## 4. Dry-run the 3D CNN plan

Run:

```powershell
train_cnn_model `
  --clips data\processed\how2sign_sentences_top12.frames.jsonl `
  --dry-run
```

You should see:

- input shape `16 x 112 x 112 x 3`
- train / val / test counts
- number of sentence classes

For the normalized `top25` experiment, use:

```powershell
train_cnn_model `
  --clips data\processed\how2sign_sentences_top25.normalized.frames.jsonl `
  --dry-run
```

## 5. Install TensorFlow training extras

Run:

```powershell
python -m pip install -e ".[training]"
```

## 6. Train the 3D CNN

Run:

```powershell
train_cnn_model `
  --clips data\processed\how2sign_sentences_top12.frames.jsonl `
  --output models\cnn-3d-sentence.keras `
  --epochs 12 `
  --batch-size 2 `
  --frame-count 16 `
  --image-size 112
```

Outputs:

```text
models/cnn-3d-sentence.keras
models/cnn-3d-sentence.labels.json
```

For the normalized `top25` experiment, use:

```powershell
train_cnn_model `
  --clips data\processed\how2sign_sentences_top25.normalized.frames.jsonl `
  --output models\cnn-3d-sentence-top25-normalized.keras `
  --epochs 20 `
  --batch-size 2 `
  --frame-count 16 `
  --image-size 112
```

## Notes

- This training path is heavier than the WLASL landmark CNN.
- Start with the top 10-12 repeated sentence classes before expanding.
- Label normalization is the safest first improvement because it does not
  require changing the 3D CNN architecture.
- If memory is tight, reduce:
  - `--batch-size` to `1`
  - `--image-size` to `96`
  - `--frame-count` to `12`

## Recommended project framing

For the report and slides, describe this branch as:

> A sentence-level closed-vocabulary 3D CNN baseline trained on repeated
> How2Sign frontal RGB clips, used to extend the project from isolated sign
> recognition toward full-sentence video classification.
