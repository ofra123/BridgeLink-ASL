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

Sentence mode now has two model branches so the team can compare CNN against VLM:

```text
camera frames
-> sentence window builder
-> sampled keyframes
-> CNN baseline OR VLM interpreter
-> comparable model result
-> transcript and speech output
```

The CNN branch receives fixed-length sampled frame stacks and predicts one of the supported sentence/gloss classes. The VLM branch receives sampled frames, and optionally the current word-token trace, then outputs a natural English sentence. Both branches use the same clip manifest so the comparison is fair.

## Why This Shape

- It keeps the demo working before the real model is trained.
- It makes the later MediaPipe and TensorFlow Lite swaps isolated rather than invasive.
- It supports offline testing by replacing camera and TTS dependencies with deterministic fallbacks.

## Phase-to-Code Mapping

- Phase 1: current scaffold, built-in seed classifier, mock landmarks, mock TTS
- Phase 2: collect sentence-window metadata and train the sampled-frame CNN baseline
- Phase 3: add a mock/local VLM sentence interpreter and comparison wrapper
- Phase 4: harden the live and backup demo paths
