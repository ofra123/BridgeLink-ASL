# Results

Place the training outputs here after running the Colab notebook.

## Files to download from Google Drive → BridgeLink-ASL → results/

| File | Used by |
|---|---|
| `metrics.json` | The app's Results tab, and for filling in the report TODOs |
| `class_distribution.png` | Report Figure 1 |
| `split_distribution.png` | Report Figure 2 |
| `training_curves.png` | Report Figure 3 |
| `confusion_matrix.png` | Report Figure 4 |
| `sample_frames.png` | Report / presentation dataset slide |
| `classification_report.txt` | Report per-class precision/recall table |

Copy these into `report/figures/` as well for the LaTeX build.

## Hybrid VLM evaluation outputs

Run:

```bash
python scripts/evaluate_hybrid_vlm.py --manifest data/vlm_eval_wlasl25/wlasl25_hybrid_eval.jsonl --output-dir results/vlm_eval
```

Expected generated files:

| File | Used by |
|---|---|
| `vlm_eval/vlm_review_template.csv` | VLM/manual reranking worksheet with prompts and blank predictions |
| `vlm_eval/vlm_hybrid_results.jsonl` | Merged eval rows, including VLM predictions after scoring |
| `vlm_eval/vlm_hybrid_metrics.json` | Transformer top-1, top-5 coverage, and VLM rerank accuracy |
