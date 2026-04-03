# Architecture Summary

BridgeLink ASL is organized around a simple perception-to-output loop:

1. `camera.py` provides frames from a webcam or a synthetic fallback source.
2. `landmarks.py` converts each frame into a landmark vector boundary.
3. `classifier.py` predicts the most likely sign from that vector.
4. `smoothing.py` stabilizes predictions across consecutive frames.
5. `translation.py` maps stable labels into readable text.
6. `speech.py` speaks the translated text with a selected provider.
7. `pipeline.py` orchestrates the full loop and writes transcript events.

## Why This Shape

- It keeps the demo working before the real model is trained.
- It makes the later MediaPipe and TensorFlow Lite swaps isolated rather than invasive.
- It supports offline testing by replacing camera and TTS dependencies with deterministic fallbacks.

## Phase-to-Code Mapping

- Phase 1: current scaffold, built-in seed classifier, mock landmarks, mock TTS
- Phase 2: replace the mock landmark path with real preprocessing and train a stronger classifier
- Phase 3: wire live inference to the trained artifact and stabilize the user-facing transcript
- Phase 4: improve reliability, startup checks, logging, and presentation workflow
