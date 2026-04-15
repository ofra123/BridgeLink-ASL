# Phase 1: Stabilize Current Repo And Word-Level Baseline

## Goal

Lock down what already works before adding sentence-level VLM behavior. The repo currently supports a word/letter-style pipeline: frame source, landmark sample, classifier label, smoothing, text output, and speech fallback. Phase 1 makes that baseline reliable, documented, and easy for every teammate to run.

## Deliverables

- remove committed virtual environments and generated artifacts from Git tracking
- keep `.venv/`, `.venv311/`, `outputs/`, `.tmp-tests/`, `models/*.json`, and large raw data out of normal commits unless explicitly approved
- verify the current `run_demo`, `train_model`, and `evaluate_model` entrypoints still work after cleanup
- keep the existing word-level pipeline as the fallback path for the final demo
- document the current limitations: single-frame or stabilized-label output, not true ASL sentence understanding

## Exit Criteria

- a fresh clone installs without pulling a committed virtual environment
- unit tests pass from a clean environment
- the word-level demo can still emit text and speech output
- the team can explain exactly what the current baseline does and does not do
- the repo is safe to build on for sentence-window and VLM work
