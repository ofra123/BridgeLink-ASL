# Phase 2: WLASL Data And CNN Baseline

## Goal

Move from placeholder demos to a real video-based classifier using WLASL and a
landmark CNN baseline.

## Deliverables

- finalize WLASL-100 as the main training dataset
- define the WLASL-25 subset used for live demo and VLM reranking
- extract fixed-length landmark sequences from labeled clips
- implement and train the temporal CNN baseline
- save model artifacts and label metadata for local and Hugging Face use
- document dataset licensing and keep large raw downloads out of Git

## Implementation Notes

- Use MediaPipe Holistic to convert each clip into a `(32, 225)` landmark tensor.
- Train the CNN baseline first, then derive the WLASL-25 demo checkpoint.
- Reuse the same WLASL-25 clips for the CNN versus VLM comparison.

## Exit Criteria

- the repo contains a stable WLASL evaluation manifest
- the CNN dry-run works locally
- the trained checkpoint loads in the Gradio app
- the team can explain why WLASL matches the final project scope
