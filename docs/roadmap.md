# BridgeLink ASL Updated Roadmap

## Current State

The repo currently has a working word-level architecture:

- webcam or synthetic frame source
- frame-to-landmark boundary
- baseline classifier that emits one sign label at a time
- smoothing to avoid repeated noisy labels
- text translation and speech fallback
- data collection support for isolated sign landmark samples

This is a good baseline, but it does not yet understand sentence-level ASL. Sentence understanding needs short video windows because meaning depends on motion, ordering, timing, facial context, and grammar.

## Local VLM Choice

The planned local VLM is `Qwen/Qwen2.5-VL-72B-Instruct`.

Reasons:

- it is the largest instruction-tuned model in the Qwen2.5-VL collection
- it supports image and video-style inputs
- it has explicit long-video and temporal event-understanding support
- it can produce structured outputs, which we need for parseable sentence events
- it can accept sampled frames plus a structured token trace from our current classifier

This is a high-hardware local model. If the team machine cannot run the 72B model smoothly, use `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` as the first fallback and `Qwen/Qwen2.5-VL-7B-Instruct` as the emergency lower-hardware fallback. The project must also keep a `mock` VLM provider for tests and offline demos.

We are not planning to depend on a cloud model for the final demo. The local model should be downloaded ahead of time, and the demo should still have a no-network fallback.

## Updated Four Phases

1. Stabilize the current repo and word-level baseline.
2. Build sentence-window data and gesture trace support.
3. Add the VLM sentence interpreter.
4. Harden the final demo and evaluation story.

The detailed phase docs live in `docs/phases`.

## Target Architecture

```text
camera frames
-> sentence window builder
-> sampled keyframes plus landmark traces
-> word-level classifier token trace
-> VLM sentence interpreter
-> sentence event
-> transcript and speech output
```

The VLM should not replace the current classifier immediately. It should use the classifier output as grounding so it produces safer, more explainable sentences.

## Demo Target

The final demo should support two flows:

- `word` mode: current stable baseline that emits recognized signs
- `sentence` mode: short clip/window input that emits a natural English sentence

The backup demo should use pre-recorded sentence clips if live signing or API access fails.
