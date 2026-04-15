# Final Project Workflow

This workflow assumes the project uses a small How2Sign subset for the final experiment and Hugging Face Space for the final proof-of-concept demo.

## 1. Create The How2Sign Subset Manifest

Download How2Sign metadata/videos outside Git, then create a BridgeLink manifest:

```powershell
python scripts/create_how2sign_subset.py `
  --metadata C:\path\to\how2sign_train.csv C:\path\to\how2sign_val.csv C:\path\to\how2sign_test.csv `
  --video-root data/raw/how2sign `
  --output data/processed/how2sign_subset.jsonl `
  --max-per-split 40
```

The manifest contains clip IDs, split labels, English targets, generated sentence labels, local video paths, and sampled frame paths.

## 2. Extract Frames And Validate The Dataset

After the videos exist locally:

```powershell
python scripts/prepare_clip_dataset.py `
  --manifest data/processed/how2sign_subset.jsonl `
  --extract-frames `
  --frame-count 16 `
  --output-manifest data/processed/how2sign_subset.jsonl `
  --summary-output results/dataset-summary.json
```

Then validate the manifest:

```powershell
python scripts/prepare_clip_dataset.py `
  --manifest data/processed/how2sign_subset.jsonl `
  --require-frames
```

## 3. Train The CNN Baseline

```powershell
pip install -e ".[training]"
train_cnn_model `
  --clips data/processed/how2sign_subset.jsonl `
  --output models/cnn-baseline.keras `
  --epochs 12 `
  --batch-size 4 `
  --frame-count 16
```

The CNN architecture, loss, optimizer, learning rate, frame count, and batch size are documented in the README and `docs/model-comparison.md`.

## 4. Run CNN vs VLM Evaluation

Until real CNN/Qwen outputs are connected, generate scaffolded report assets:

```powershell
python scripts/generate_project_results.py `
  --manifest data/processed/how2sign_subset.jsonl `
  --output-dir results
```

After real model inference is available, replace the scaffolded predictions with actual CNN and Qwen2.5-VL outputs while keeping the same JSON/JSONL artifact names.

## 5. Use The Space As The Final Dashboard

The Hugging Face Space displays:

- live upload/webcam demo
- How2Sign subset summary
- class and split charts
- CNN vs VLM metrics
- confusion matrix
- CVPR report checklist

This makes the Space the demo and project dashboard, while the report contains the formal technical explanation and results.
