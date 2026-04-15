# Hugging Face Space Proof Of Concept

## Feasibility

Yes, BridgeLink ASL can run as a proof of concept on Hugging Face Spaces.

The hosted final product should be scoped as:

- browser UI with upload and webcam-recorded clips
- CNN-style baseline output
- VLM-style sentence output
- side-by-side comparison, confidence, latency, and limitations
- reliable fallback behavior when no GPU model is attached

The full research-grade version still needs:

- Trey: real How2Sign/team clip data and sampled frame paths
- Omar: trained CNN artifact from those frames
- Frank: real Qwen2.5-VL runtime on GPU hardware
- Dalen: final benchmark wrapper and metric export

## What Runs Now

The Space branch now has an end-to-end hosted PoC:

```text
upload/webcam clip
-> OpenCV video feature extraction
-> CNN-style hosted baseline
-> VLM-style grounded sentence output
-> comparison report
-> final sentence / speech-ready text
```

If OpenCV cannot read the video, the app falls back to file metadata instead of crashing. If no video is provided, the app returns a safe demo sentence and reports that fallback state.

## Why This Counts As A PoC

The app proves the final user flow:

- a user can record a live webcam clip in the browser
- the system accepts the clip
- the system runs both comparison branches
- the system returns a sentence-level output
- the system exposes model status, confidence, latency, and limitations

It does not yet prove model accuracy. Accuracy requires real data, real training, and real Qwen inference.

## Hosted Model Strategy

The default VLM target remains:

```text
Qwen/Qwen2.5-VL-32B-Instruct-AWQ
```

This is still large. For a free or CPU Space, keep the current grounded mock VLM. For a GPU Space, Frank can replace the mock VLM function with a real Qwen provider while preserving the same JSON output contract.

## Demo Script

1. Open the Space.
2. Choose `Run Demo` or `Webcam Only`.
3. Record a 2-5 second signing clip.
4. Select `Compare`.
5. Click `Run BridgeLink ASL`.
6. Explain that the hosted PoC proves the pipeline and UI; the next milestone replaces the fallback CNN/VLM branches with trained/runtime models.
