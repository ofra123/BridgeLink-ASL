# Hugging Face Space Proof Of Concept

## Feasibility

Yes, BridgeLink ASL runs as a proof of concept on Hugging Face Spaces.

The hosted final product is scoped as:

- browser UI with upload and webcam-recorded clips
- landmark CNN output
- live tracking overlay
- confidence-aware captioning
- reliable fallback behavior when webcam or model input is weak

The full research workflow still lives partly off-Space:

- WLASL training and landmark extraction are done locally or in Colab/Kaggle
- local Qwen2.5-VL reranking is used for comparison experiments
- only the CNN demo path is expected to be practical on a free CPU Space

## What Runs Now

The Space hosts an end-to-end CNN demo:

```text
upload/webcam clip
-> MediaPipe landmark extraction
-> temporal CNN inference
-> smoothing and caption output
-> presentation-ready demo UI
```

## Why This Counts As A PoC

The app proves the final user flow:

- a user can record a live webcam clip in the browser
- the system accepts the clip
- the system runs the trained CNN branch
- the system returns a word-level ASL prediction
- the system exposes model status and a tracking overlay

Accuracy claims come from the offline WLASL experiments and the local VLM
comparison workflow rather than from the Space itself.

## Demo Script

1. Open the Space.
2. Test `Upload / Record Clip` first.
3. Then test `Live Webcam`.
4. Show the overlay, candidate label, and caption output.
5. Explain that the Space is the live CNN demo surface, while the VLM comparison runs offline.
