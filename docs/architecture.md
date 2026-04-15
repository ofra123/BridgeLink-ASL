# Architecture Summary

BridgeLink ASL is organized around a simple perception-to-output loop:

1. `camera.py` provides frames from a webcam or a synthetic fallback source.
2. `landmarks.py` converts each frame into a landmark vector boundary.
3. `classifier.py` predicts the most likely sign from that vector.
4. `smoothing.py` stabilizes predictions across consecutive frames.
5. `translation.py` maps stable labels into readable text.
6. `speech.py` speaks the translated text with a selected provider.
7. `pipeline.py` orchestrates the full loop and writes transcript events.

## Next Architecture: Sentence Mode

Sentence mode adds a layer above the current word-level baseline:

```text
camera frames
-> sentence window builder
-> sampled keyframes and landmark traces
-> word-level token trace
-> VLM sentence interpreter
-> sentence event
-> transcript and speech output
```

The VLM should receive both visual context and structured classifier output. This keeps the sentence output grounded and makes it easier to explain mistakes during the demo.

## Why This Shape

- It keeps the demo working before the real model is trained.
- It makes the later MediaPipe and TensorFlow Lite swaps isolated rather than invasive.
- It supports offline testing by replacing camera and TTS dependencies with deterministic fallbacks.

## Phase-to-Code Mapping

- Phase 1: current scaffold, built-in seed classifier, mock landmarks, mock TTS
- Phase 2: collect sentence-window metadata and create token traces
- Phase 3: add a mock and cloud VLM sentence interpreter
- Phase 4: harden the live and backup demo paths
