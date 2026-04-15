# BridgeLink ASL

BridgeLink ASL is a Python-first proof of concept for real-time American Sign Language recognition, text translation, and speech output. This repo is intentionally scaffolded to support the four class-project phases without locking the team into a web stack too early.

The current starter build focuses on:

- a runnable webcam-demo shell with synthetic fallback
- a hand-landmark style pipeline boundary that can later swap to MediaPipe
- a lightweight centroid baseline for training and evaluation
- optional speech adapters with mock fallback
- docs and phase artifacts that keep the team aligned

## Quick Start

Create a virtual environment and install the starter package:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Run the starter demo:

```powershell
run_demo --frames 18
```

Train the baseline centroid model against the sample landmark dataset:

```powershell
train_model --dataset data/processed/sample_landmarks.jsonl --output models/dev-baseline.json
```

Evaluate the saved model:

```powershell
evaluate_model --dataset data/processed/sample_landmarks.jsonl --model models/dev-baseline.json
```

The demo works even before a trained model exists. If `models/dev-baseline.json` is missing, the pipeline falls back to the built-in seed classifier so the end-to-end flow remains usable during phase 1.

## Repo Layout

- `src/bridgelink_asl`: app package, runtime pipeline, training, and evaluation logic
- `tests`: unit and smoke tests for config, data validation, smoothing, translation, speech selection, and the demo loop
- `data`: dataset guidance plus a small sample landmark dataset
- `models`: saved baseline artifacts
- `docs`: architecture notes, V1 scope, demo checklist, and the four phase documents

## Phase Overview

1. Phase 1: stabilize the current repo and word-level baseline
2. Phase 2: build sentence-window data and gesture trace support
3. Phase 3: add the VLM sentence interpreter
4. Phase 4: harden the demo, evaluation, and presentation flow

Detailed phase documents live in `docs/phases`. The updated roadmap is in `docs/roadmap.md`, local VLM guidance is in `docs/local-vlm.md`, and dataset guidance is in `docs/datasets.md`.

## V1 Sign Scope

The starter vocabulary intentionally targets a limited high-value sign set so the demo can be reliable:

- `HELLO`
- `YES`
- `NO`
- `THANK_YOU`
- `PLEASE`
- `HELP`
- `STOP`
- `EAT`
- `DRINK`
- `WANT`
- `MORE`
- `FINISHED`

See `docs/v1-scope.md` for definitions and phase-1 success criteria.

## Dependency Strategy

The starter repo is stdlib-first so it runs cleanly in a class environment. Optional packages are grouped in `pyproject.toml`:

- `demo`: OpenCV and MediaPipe
- `speech`: local text-to-speech
- `training`: TensorFlow-based follow-on work
- `cloud`: ElevenLabs

This keeps phase 1 lightweight while leaving clear extension points for phases 2 through 4.
