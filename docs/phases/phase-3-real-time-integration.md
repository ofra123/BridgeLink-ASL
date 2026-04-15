# Phase 3: VLM Sentence Interpreter Integration

## Goal

Add a VLM-backed sentence interpreter that turns a short gesture window plus detected sign tokens into a natural English sentence. The VLM should be grounded by the token trace and should not be allowed to invent signs freely.

## Deliverables

- add a `SentenceInterpreter` interface with mock and cloud-backed implementations
- use `Qwen/Qwen2.5-VL-72B-Instruct` as the planned local VLM, with `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` as the first fallback and `Qwen/Qwen2.5-VL-7B-Instruct` as the emergency lower-hardware fallback
- add `GestureWindow`, `DetectedGestureToken`, and `SentenceEvent` types
- update the pipeline so it supports two modes: `word` and `sentence`
- use the current classifier and smoother to produce token traces for each sentence window
- design a strict JSON prompt that gives the VLM sampled frames, detected tokens, confidence scores, and expected output fields
- require a mock/local interpreter for tests and offline demos
- add confidence and fallback behavior: if the VLM is uncertain, use the token trace as a simple gloss sentence instead of hallucinating

## Exit Criteria

- sentence mode can process a recorded or synthetic gesture window
- the mock VLM interpreter passes tests without network access or API keys
- local VLM mode is optional and controlled by config
- transcript output includes both raw tokens and the final sentence
- speech output reads the final sentence, not every individual word
