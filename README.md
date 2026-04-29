---
title: BridgeLink ASL
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---

# BridgeLink ASL

Real-time word-level American Sign Language recognition using MediaPipe
Holistic landmarks and a lightweight CNN classifier, trained on WLASL-100 and
compared against a zero-shot VLM reranking workflow.

## Demo

The Gradio app supports three modes:

- **Live Webcam** — streaming video with a rolling 32-frame sliding window,
  near-real-time sign prediction, and caption display.
- **Upload / Record Clip** — upload or record a 2–5 second clip, get top-5
  predictions with confidence scores.
- **How2Sign Sentence CNN** — upload a short RGB clip and classify it with the
  closed-vocabulary sentence-level 3D CNN trained on repeated How2Sign
  sentences.

For a full local setup guide, see
[`docs/local-machine-setup.md`](docs/local-machine-setup.md).

### Run locally

```bash
py -3.11 -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m pip install -e .
set HF_MODEL_FILENAME=cnn_landmark_wlasl25_best.pt
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

### Hugging Face Space

The same app runs on a free CPU Space.

For the landmark live demo, set:

- `HF_MODEL_REPO` to a HF model repo containing `cnn_landmark_wlasl25_best.pt`
- optionally `HF_MODEL_FILENAME=sign_transformer_best.pt` if you want to demo
  the Transformer checkpoint instead

For the How2Sign sentence tab, set:

- `HF_SENTENCE_MODEL_REPO` to a HF model repo containing
  `cnn-3d-sentence-top25-normalized.keras`
- optionally `HF_SENTENCE_MODEL_FILENAME` if you publish the sentence model
  under a different filename

If you keep the model artifacts directly in the Space repo under `models/`,
the app will load them locally without any HF Hub environment variables.

## Method

```
webcam frame (30 FPS)
  -> MediaPipe Holistic -> 225-d landmark vector
  -> rolling buffer of 32 frames (~1 second)
  -> temporal 1D CNN over the landmark sequence
  -> top-1 sign + confidence
  -> stability filter (2 consecutive agreeing predictions, confidence >= 0.35)
  -> live caption + optional TTS
```

The optional Transformer checkpoint uses the same landmark tensors with a
CLS-token attention encoder and is kept as an extra attention experiment.

## Dataset

[WLASL-100](https://dxli94.github.io/WLASL/) — the 100 most frequent glosses
from the Word-Level American Sign Language video dataset (Li et al., WACV 2020).
Licensed under the Computational Use of Data Agreement (C-UDA).

## Training

The training notebook is at `notebooks/train_wlasl100_colab.ipynb`. It runs on
Google Colab (free T4 GPU), trains the required landmark CNN, optionally trains
the Transformer extension, and saves all outputs to Google Drive.

## Hybrid VLM Evaluation

After the notebook creates
`vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl`, copy that folder into the
repo or point the script at the downloaded file:

```bash
python scripts/evaluate_hybrid_vlm.py ^
  --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl ^
  --output-dir results/vlm_eval
```

This writes a baseline metrics file plus `vlm_review_template.csv`. Fill the
`vlm_prediction` column using a VLM that chooses from the CNN's top-5
candidate labels only, then rescore:

```bash
python scripts/evaluate_hybrid_vlm.py ^
  --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl ^
  --predictions results/vlm_eval/vlm_review_template.csv ^
  --output-dir results/vlm_eval
```

Report the three comparison numbers: CNN top-1 accuracy, CNN top-5 coverage,
and VLM-reranked top-5 accuracy. The Transformer result is kept as an optional
attention/extra-credit experiment.

## Imported How2Sign VLM Workspace

The teammate Qwen2.5-VL fine-tuning workspace is now included directly in this
repo under `vlm_hf_space/`.

That folder preserves:

- QLoRA training scripts
- How2Sign/ASL Citizen prep scripts
- experiment configs
- archived baseline vs fine-tuned metrics
- sample prediction outputs from the Hugging Face Space repo

The original source repo was:
`https://huggingface.co/spaces/ofraij123/ASL-Video-To-Sentence-Translation`

## Wrapper Scaffold

The repo also includes a sentence-wrapper CLI for phase-3 style comparisons
over clip manifests:

```bash
run_wrapper --mode compare --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl
```

The mock interpreter still works offline for testing, and the local Qwen path
now loads lazily when the optional VLM extras are installed:

```bash
pip install -e ".[vlm]"
run_wrapper --mode compare --vlm-provider local --manifest data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl
```

The wrapper accepts both the older `clip_id`/`gloss` sentence manifests and the
current hybrid WLASL eval manifests with `video_id`, `true_label`, and
`cnn_top5`.

## Repo Layout

```
app.py                          Gradio Space entrypoint
requirements.txt                Space / local dependencies
src/bridgelink_asl/
  inference.py                  CNN/Transformer loader + MediaPipe extraction runtime
  sentence_inference.py         How2Sign sentence CNN loader + clip preprocessing
  wrapper.py                    Sentence wrapper scaffold for cnn/vlm/compare runs
  (other modules)               Supporting pipeline code
models/
  cnn_landmark_best.pt          Primary CNN weights (download from Drive after training)
  cnn_landmark_wlasl25_best.pt  Smaller live-demo CNN weights
  cnn-3d-sentence-top25-normalized.keras   Best How2Sign repeated-sentence 3D CNN weights
  cnn-3d-sentence-top25-normalized.labels.json
  sign_transformer_best.pt      Optional Transformer weights
  labels.json                   Label map
results/
  metrics.json                  Test accuracy, top-5, etc.
  *.png                         Plots for the report
notebooks/
  train_wlasl100_colab.ipynb    End-to-end training notebook
report/
  main.tex                      CVPR-style project report
  references.bib                Bibliography
vlm_hf_space/                   Imported Qwen2.5-VL workspace from Hugging Face Space
docs/                           Architecture notes, phase plans, literature review
tests/                          Unit tests
```

## Team

| Name | Role | Email |
|---|---|---|
| Dalen Gordon | Undergraduate | dgordo34@charlotte.edu |
| Ervin Gordon III | Undergraduate | egordo17@charlotte.edu |
| Frank Garcia | Graduate | fgarci11@charlotte.edu |
| Omar Fraij | Undergraduate | ofraij@charlotte.edu |

ITCS 4152/5010 — Introduction to Computer Vision, Spring 2026
University of North Carolina at Charlotte
