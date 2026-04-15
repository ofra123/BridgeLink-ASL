# Local VLM Selection

## Chosen Model

Use `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` as the primary 30B-class VLM for sentence mode and Hugging Face Space experiments.

Why this model fits BridgeLink ASL:

- it is the practical 32B AWQ quantized target from the Qwen2.5-VL collection
- it supports visual and video-style inputs
- it is designed for long-video/event understanding, which matters for ASL sentence windows
- it can return structured text that we can parse into sentence events
- it can consume sampled keyframes plus the current word-level token trace

## Hardware Reality And Fallbacks

The 32B AWQ model is still large. It is more realistic than 72B, but it may still require paid Hugging Face Space hardware, quantization-friendly inference, or a strong local GPU setup.

Fallback order:

- `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` as the current default
- `Qwen/Qwen2.5-VL-7B-Instruct` if the team needs a laptop-friendly emergency path
- `Qwen/Qwen2.5-VL-72B-Instruct` as a high-hardware stretch comparison if available
- `MockSentenceInterpreter` for tests and no-network/no-model demos

## Integration Plan

Start with a provider interface:

```text
SentenceInterpreter
-> MockSentenceInterpreter
-> LocalQwen25VlmInterpreter
```

The mock interpreter should always work without model downloads. The local Qwen2.5-VL interpreter should be enabled only when the model is installed and configured.

The VLM path is now compared directly against the CNN baseline. It should consume the same clip manifest records as the CNN branch, then produce sentence-level JSON that the comparison wrapper can score.

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

Download the model before presentation day. The final demo should not depend on internet access. Also keep a recorded-clip fallback because 32B local inference can still be too slow for a live webcam demo on weak hardware.
