# Presentation Visuals

Generated slide-ready visuals for BridgeLink ASL.

## Files

- `bridgelink_pipeline.svg`: methodology / system diagram
- `project_scope_board.svg`: what the team trained versus what was reused
- `cnn_vs_vlm_comparison.svg`: final comparison numbers for the presentation
- `metrics-summary.json`: source values used to render the comparison slide

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
```
