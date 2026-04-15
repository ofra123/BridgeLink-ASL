# Final Project Workflow

This branch makes the final professor-facing comparison explicit:

```text
trained landmark CNN vs zero-shot VLM reranking
```

The Transformer remains in the notebook as an optional attention/extra-credit
experiment, not the main comparison.

## 1. Train From The Colab Notebook

Use `notebooks/train_wlasl100_colab.ipynb`.

The notebook expects:

```text
/content/drive/MyDrive/BridgeLink-ASL/data/wlasl-processed.zip
```

Run the notebook top-to-bottom. If landmarks are already extracted, Step 7 is
cached and should report mostly `Cached` samples instead of redoing the slow
MediaPipe pass.

## 2. Primary CNN Outputs

The CNN section saves:

```text
/content/drive/MyDrive/BridgeLink-ASL/models/cnn_landmark_best.pt
/content/drive/MyDrive/BridgeLink-ASL/results/cnn_metrics.json
/content/drive/MyDrive/BridgeLink-ASL/results/cnn_training_curves.png
/content/drive/MyDrive/BridgeLink-ASL/results/cnn_confusion_matrix.png
/content/drive/MyDrive/BridgeLink-ASL/results/cnn_classification_report.txt
```

These are the main model artifacts for the report.

## 3. Optional Transformer Outputs

The Transformer cells are kept after the CNN section. Run them only if time
allows and report them as an attention-based extension.

```text
/content/drive/MyDrive/BridgeLink-ASL/models/sign_transformer_best.pt
/content/drive/MyDrive/BridgeLink-ASL/results/metrics.json
```

## 4. Build The CNN/VLM Eval Set

The CNN/VLM notebook cell creates:

```text
/content/drive/MyDrive/BridgeLink-ASL/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl
/content/drive/MyDrive/BridgeLink-ASL/vlm_eval_wlasl25_cnn/clips/
```

Each row includes the true label, clip path, CNN top-1, CNN top-5 candidates,
and a constrained prompt for the VLM.

## 5. Score The VLM Reranker

Copy the eval folder into:

```text
data/vlm_eval_wlasl25_cnn/
```

Generate the review sheet and baseline metrics:

```powershell
python scripts/evaluate_hybrid_vlm.py `
  --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl `
  --output-dir results/vlm_eval
```

Fill `results/vlm_eval/vlm_review_template.csv` with VLM choices, then rescore:

```powershell
python scripts/evaluate_hybrid_vlm.py `
  --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl `
  --predictions results/vlm_eval/vlm_review_template.csv `
  --output-dir results/vlm_eval
```

Report:

```text
CNN top-1 accuracy
CNN top-5 coverage
VLM-reranked top-5 accuracy
```

## 6. Deploy The Space

Upload `cnn_landmark_best.pt` to a Hugging Face model repo and set Space
variables:

```text
HF_MODEL_REPO=<username>/<model-repo>
HF_MODEL_FILENAME=cnn_landmark_best.pt
```

The Space can still load `sign_transformer_best.pt` if you set
`HF_MODEL_FILENAME=sign_transformer_best.pt`, but the final demo should use the
CNN checkpoint unless the professor asks for the Transformer extension.
