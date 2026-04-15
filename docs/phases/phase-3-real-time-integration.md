# Phase 3: VLM Integration And Comparison Wrapper

## Goal

Add a VLM-backed sentence interpreter and a wrapper that compares VLM output against the CNN baseline on the same clips. The VLM should use visual context, and optionally token traces, while the CNN stays the controlled baseline.

## Deliverables

- add a `SentenceInterpreter` interface with mock and cloud-backed implementations
- use `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` as the planned 30B-class VLM, with `Qwen/Qwen2.5-VL-7B-Instruct` as the emergency lower-hardware fallback and `Qwen/Qwen2.5-VL-72B-Instruct` as a high-hardware stretch target
- add `GestureWindow`, `DetectedGestureToken`, and `SentenceEvent` types
- update the pipeline so it supports `word`, `cnn`, `vlm`, and `compare` modes
- use the current classifier and smoother to produce token traces for each sentence window
- design a strict JSON prompt that gives the VLM sampled frames, detected tokens, confidence scores, and expected output fields
- require a mock/local interpreter for tests and offline demos
- add confidence and fallback behavior: if the VLM is uncertain, use the token trace as a simple gloss sentence instead of hallucinating
- add comparison output with clip ID, expected text, CNN prediction, VLM prediction, latency, and failure notes

## Owners

- Dalen: wrapper modes, shared input contract, comparison logs
- Frank: local Qwen2.5-VL provider, prompt, structured VLM output

## Exit Criteria

- sentence mode can process a recorded or synthetic gesture window
- compare mode can run CNN and VLM predictions over the same manifest
- the mock VLM interpreter passes tests without network access or API keys
- local VLM mode is optional and controlled by config
- transcript output includes both raw tokens and the final sentence
- speech output reads the final sentence, not every individual word
