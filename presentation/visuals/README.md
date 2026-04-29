# Presentation Visuals

Generated slide-ready visuals for BridgeLink ASL.

## Files

- `bridgelink_pipeline.svg`: methodology / system diagram
- `project_scope_board.svg`: what the team trained versus what was reused
- `cnn_vs_vlm_comparison.svg`: final comparison numbers for the presentation
- `metrics-summary.json`: source values used to render the comparison slide
- `how2sign_dataset_constraint.(png|svg)`: explains why the full 31k How2Sign clips cannot be used as sentence classes
- `how2sign_subset_benchmark.(png|svg)`: Top-12, Top-25, and Top-25 normalized benchmark growth and accuracy
- `how2sign_top25_class_distribution.(png|svg)`: class imbalance view for the normalized repeated-sentence subset
- `how2sign_top25_experiment_comparison.(png|svg)`: comparison of the Top-25 baseline, normalized checkpoint, continued training, and balanced-weight experiments
- `how2sign_plot_metrics.json`: source values used by the matplotlib How2Sign plots

## Current values

- Evaluation set: 36 clips
- Unique classes: 22
- CNN top-1: 25.0%
- CNN top-5: 58.3%
- Qwen rerank: 25.0%
- VLM wrapper failures: 0

## Regenerate

```powershell
python scripts\generate_presentation_visuals.py
python scripts\generate_how2sign_presentation_plots.py
```

## Current How2Sign 3D CNN values

- Top-12 test accuracy: 14.7%
- Top-25 test accuracy: 20.8%
- Top-25 normalized test accuracy: 27.8%
- Top-25 normalized classes: 21
