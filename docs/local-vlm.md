# Local VLM Selection

## Chosen Model

Use `Qwen/Qwen2.5-VL-72B-Instruct` as the primary local VLM for sentence mode.

Why this model fits BridgeLink ASL:

- it is the 72B instruction-tuned model from the Qwen2.5-VL collection
- it supports visual and video-style inputs
- it is designed for long-video/event understanding, which matters for ASL sentence windows
- it can return structured text that we can parse into sentence events
- it can consume sampled keyframes plus the current word-level token trace

## Hardware Reality And Fallbacks

The 72B model is powerful but large. It is the target model, but it may require a strong GPU setup, quantization, or multi-GPU hardware to run smoothly.

Fallback order:

- `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` if the full 72B model is too heavy
- `Qwen/Qwen2.5-VL-7B-Instruct` if the team needs a laptop-friendly emergency path
- `MockSentenceInterpreter` for tests and no-network/no-model demos

## Integration Plan

Start with a provider interface:

```text
SentenceInterpreter
-> MockSentenceInterpreter
-> LocalQwen25VlmInterpreter
```

The mock interpreter should always work without model downloads. The local Qwen2.5-VL interpreter should be enabled only when the model is installed and configured.

## Prompting Rule

The local VLM should receive both:

- sampled frames from a short gesture window
- the current classifier token trace, including labels, confidence, and timestamps

The model should return strict JSON:

```json
{
  "gloss": ["HELLO", "WANT", "DRINK"],
  "sentence": "Hello, I want a drink.",
  "confidence": 0.82,
  "needs_clarification": false
}
```

If the VLM output is invalid or low confidence, the pipeline should fall back to a simple gloss sentence from the token trace.

## Practical Demo Note

Download the model before presentation day. The final demo should not depend on internet access. Also keep a recorded-clip fallback because 72B local inference can be too slow for a live webcam demo on weak hardware.
