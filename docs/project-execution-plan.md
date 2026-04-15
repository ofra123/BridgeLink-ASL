# Project Execution Plan

This plan treats BridgeLink ASL as one integrated project rather than a role-split handoff.

## Goal

Complete a Hugging Face Space proof of concept backed by a reproducible How2Sign subset experiment:

```text
How2Sign subset
-> sampled video frames
-> CNN baseline
-> Qwen2.5-VL comparison
-> metrics, charts, Space demo, CVPR-style report
```

## Required Work

1. Create `data/processed/how2sign_subset.jsonl` from downloaded How2Sign metadata.
2. Extract 16 sampled frames per clip into local `data/interim/frames/`.
3. Train the CNN baseline with `train_cnn_model`.
4. Run Qwen2.5-VL-7B or precomputed Qwen outputs on the same test clips.
5. Generate `results/` artifacts with accuracy, precision, recall, F1, confusion matrix, and comparison rows.
6. Use the Hugging Face Space as the live demo and results dashboard.
7. Write the CVPR-style report from `docs/final-project-workflow.md`, `docs/literature-review.md`, and the generated results.

## Done Criteria

- Space demo runs with webcam/uploaded clips.
- Dataset tab shows How2Sign subset summary and charts.
- Experiments tab shows CNN vs VLM metrics and confusion matrix.
- Report tab summarizes the CVPR-style report structure.
- Repo includes scripts to regenerate the dataset summary and results assets.
- Final report includes dataset, methodology, hyperparameters, experiment setup, metrics, graphs, and limitations.
