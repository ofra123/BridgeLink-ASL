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

BridgeLink ASL is a computer vision project for American Sign Language video understanding. The final project compares three paths:

- a word-level landmark CNN for live webcam sign recognition
- a sentence-level 3D CNN trained on How2Sign RGB clips
- an imported Qwen2.5-VL workspace for VLM-based sentence translation experiments

## Important clarification

- The final full-sentence classifier in this repo uses **How2Sign**, not WLASL-100.
- The sentence classifier is a **3D CNN over RGB video frames**, not the older 1D landmark CNN.
- The landmark CNN remains in the repo only for the **word-level live webcam demo** and isolated-sign experiments.
- If you are reviewing the final sentence-classification work, the main model to look at is the **How2Sign 3D CNN**.

## What the app does

The Gradio app supports three modes:

- **Live Webcam** - real-time word-level ASL recognition from MediaPipe Holistic landmarks.
- **Upload / Record Clip** - upload a short sign clip and run the word-level recognition branch.
- **How2Sign Sentence CNN** - upload a short RGB sentence clip and classify it with the closed-vocabulary 3D CNN trained on repeated How2Sign sentences.

For the final class project, the sentence branch is the key contribution because it operates on **full ASL sentence clips**, not still images and not isolated single-sign labels.

## Final sentence-classification path

The sentence classifier works as follows:

```text
How2Sign RGB sentence clip
  -> decode video
  -> uniformly sample 16 frames
  -> resize frames to 112 x 112
  -> stack into a 3D video volume
  -> Conv3D + BatchNorm + MaxPool blocks
  -> GlobalAveragePooling3D
  -> Dense softmax classifier
  -> best sentence label from the repeated-sentence How2Sign subset
```

Key details:

- Dataset: **How2Sign**
- Input: **16 RGB frames per clip**
- Frame size: **112 x 112**
- Model family: **3D CNN**
- Task: **closed-vocabulary sentence classification**

The final best sentence model is the normalized top-25 How2Sign repeated-sentence model, which reduces duplicate punctuation and wording variants into 21 usable sentence classes.

## Dataset summary

### How2Sign sentence data

How2Sign is the dataset used for the sentence classifier. It provides RGB videos aligned to English sentence translations. In this project, the 3D CNN uses short frontal-view RGB sentence clips.

Important limitation:

- The full local How2Sign inventory contains about **31k clips**, but most sentence labels appear only once.
- Because a standard softmax classifier needs repeated examples per class, the 3D CNN was trained on a **repeated-sentence subset** rather than treating all 31k clips as separate classes.
- This means the final 3D CNN is a **sentence classifier over repeated sentence labels**, not an open-ended translator over every unique How2Sign sentence.

### WLASL word-level data

WLASL-based data are kept only for the **word-level webcam / isolated-sign branch**. They are **not** the dataset used for the final sentence classifier.

## VLM workspace

The teammate Qwen2.5-VL fine-tuning workspace is included directly in this repo under `vlm_hf_space/`.

That folder preserves:

- QLoRA training scripts
- How2Sign and ASL Citizen data-prep scripts
- experiment configs
- archived baseline vs fine-tuned metrics
- sample prediction outputs from the Hugging Face Space repo

Original source:

`https://huggingface.co/spaces/ofraij123/ASL-Video-To-Sentence-Translation`

## Run locally

```bash
py -3.11 -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m pip install -e .
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

## Hugging Face Space configuration

For the word-level live demo:

- `HF_MODEL_REPO` should point to a model repo containing `cnn_landmark_wlasl25_best.pt`
- optionally set `HF_MODEL_FILENAME` if you publish that model under a different name

For the sentence-level How2Sign demo:

- `HF_SENTENCE_MODEL_REPO` should point to a model repo containing `cnn-3d-sentence-top25-normalized.keras`
- optionally set `HF_SENTENCE_MODEL_FILENAME` if you publish the sentence model under a different name

If model artifacts already live under `models/` in the Space repo, the app can load them locally without additional Hugging Face Hub variables.

## Training notes

### Word-level landmark branch

The older webcam branch uses:

- MediaPipe Holistic landmark extraction
- a rolling 32-frame sequence
- a temporal 1D CNN over landmark vectors

This branch is useful for the live demo, but it is **not** the final sentence-classification pipeline.

### Sentence-level 3D CNN branch

The main sentence training entrypoint is:

```bash
python -m bridgelink_asl.cli.train_cnn_model ^
  --clips data\processed\how2sign_sentences_top25_normalized.frames.jsonl ^
  --output models\cnn-3d-sentence-top25-normalized.keras ^
  --batch-size 2 ^
  --frame-count 16 ^
  --image-size 112
```

This branch trains directly on **How2Sign sentence videos** and is the model that should be cited when describing the repo's final sentence classifier.

## Repo layout

```text
app.py                          Gradio and Hugging Face Space entrypoint
requirements.txt                Space / local dependencies
src/bridgelink_asl/
  inference.py                  Word-level landmark inference runtime
  sentence_inference.py         How2Sign sentence 3D CNN inference runtime
  wrapper.py                    CNN / VLM comparison wrapper utilities
models/
  cnn_landmark_best.pt          Word-level landmark CNN weights
  cnn_landmark_wlasl25_best.pt  Smaller word-level live-demo weights
  cnn-3d-sentence-top25-normalized.keras
                                Final How2Sign sentence 3D CNN weights
  cnn-3d-sentence-top25-normalized.labels.json
                                Sentence label map
  sign_transformer_best.pt      Optional word-level attention experiment
results/
  *.json                        Metrics and dataset summaries
  *.png                         Plots for the report and presentation
notebooks/
  train_wlasl100_colab.ipynb    Word-level landmark training notebook
report/
  main.tex                      Final report source
  BridgeLink_ASL_Final_Report.pdf
                                Final compiled report
vlm_hf_space/                   Imported Qwen2.5-VL workspace from Hugging Face Space
docs/                           Setup notes, architecture notes, and planning docs
tests/                          Unit tests
```

## Final project focus

To avoid confusion:

- **Word-level recognition** in this repo is the landmark CNN branch.
- **Full-sentence classification** in this repo is the **How2Sign 3D CNN** branch.
- **Open-ended sentence translation experiments** in this repo are in the imported **VLM workspace**.

If you need the final sentence model for the class project discussion, use the How2Sign 3D CNN path and not the older WLASL landmark branch.

## Team

| Name | Role |
|---|---|
| Dalen Gordon | Undergraduate |
| Ervin Gordon III | Undergraduate |
| Frank Garcia | Graduate |
| Omar Fraij | Undergraduate |

ITCS 4152/5010 - Introduction to Computer Vision, Spring 2026  
University of North Carolina at Charlotte
