# Project Execution Plan

This plan treats BridgeLink ASL as one integrated project rather than a role
handoff.

## Goal

Complete a Hugging Face Space proof of concept backed by reproducible WLASL
experiments:

```text
WLASL-100 training
-> MediaPipe landmark extraction
-> temporal CNN baseline
-> WLASL-25 demo checkpoint
-> Qwen2.5-VL reranking comparison
-> metrics, charts, Space demo, CVPR-style report
```

## Required Work

1. Train the landmark CNN on WLASL-100.
2. Export the smaller WLASL-25 demo checkpoint for presentation use.
3. Build the WLASL-25 hybrid eval manifest with CNN top-5 candidates.
4. Run local Qwen2.5-VL reranking on the same clip set.
5. Generate `results/` artifacts with accuracy, precision, recall, F1, confusion matrix, and comparison rows.
6. Use Hugging Face Space as the live demo surface for the CNN app.
7. Write the CVPR-style report from the docs and generated results.

## Done Criteria

- Space demo runs with webcam and uploaded clips.
- Dataset visuals reflect the WLASL evaluation subset.
- Experiments show CNN versus VLM metrics honestly.
- Repo includes scripts to regenerate metrics and presentation assets.
- Final report includes dataset, methodology, hyperparameters, experiment setup, metrics, graphs, and limitations.
