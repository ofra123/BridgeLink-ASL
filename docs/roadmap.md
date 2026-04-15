# BridgeLink ASL Updated Roadmap

## Current State

The repo currently has a working word-level architecture:

- webcam or synthetic frame source
- frame-to-landmark boundary
- baseline classifier that emits one sign label at a time
- smoothing to avoid repeated noisy labels
- text translation and speech fallback
- data collection support for isolated sign landmark samples

This is a good baseline, but it does not yet satisfy the professor-requested model comparison. The next version needs two comparable paths over the same sentence clips: a CNN baseline and a VLM interpretation path.

The LSTM/landmark path is now a possible stretch or support tool, not the core comparison. For the graded comparison, the two headline models are:

- CNN baseline: sampled video frames from a clip -> sentence/gloss class
- VLM path: sampled frames plus optional token trace -> natural English sentence

## Local VLM Choice

The planned 30B-class VLM is `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`.

Reasons:

- it is a practical 32B AWQ quantized Qwen2.5-VL target
- it supports image and video-style inputs
- it has explicit long-video and temporal event-understanding support
- it can produce structured outputs, which we need for parseable sentence events
- it can accept sampled frames plus a structured token trace from our current classifier

This is still a high-hardware model. If the team machine or Hugging Face Space cannot run the 32B AWQ model smoothly, use `Qwen/Qwen2.5-VL-7B-Instruct` as the emergency lower-hardware fallback. If stronger hardware is available, the original `Qwen/Qwen2.5-VL-72B-Instruct` target can remain a stretch comparison. The project must also keep a `mock` VLM provider for tests and offline demos.

We are not planning to depend on a cloud model for the final demo. The local model should be downloaded ahead of time, and the demo should still have a no-network fallback.

## Updated Four Phases

1. Stabilize the current repo and word-level baseline.
2. Build sentence-window data and the CNN baseline.
3. Add the VLM sentence interpreter and comparison wrapper.
4. Harden the final demo and evaluation story.

The detailed phase docs live in `docs/phases`.

## Target Architecture

```text
camera frames
-> sentence window builder
-> sampled keyframes
-> CNN baseline OR VLM sentence interpreter
-> comparable model result
-> transcript and speech output
```

The CNN and VLM must read the same clip manifest so accuracy, latency, and failure cases can be compared fairly. The existing word classifier can still provide a token trace as grounding for the VLM, but it is not the main model being compared.

## Demo Target

The final demo should support two flows:

- `word` mode: current stable baseline that emits recognized signs
- `cnn` mode: short clip/window input classified by the sampled-frame CNN
- `vlm` mode: short clip/window input interpreted by Qwen2.5-VL
- `compare` mode: run CNN and VLM on the same clip manifest and report accuracy/latency/failure modes

The backup demo should use pre-recorded sentence clips if live signing or API access fails.
