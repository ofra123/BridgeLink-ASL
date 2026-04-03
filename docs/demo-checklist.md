# Demo Checklist

## Primary Flow

1. Activate the virtual environment and install the package.
2. Train or refresh `models/dev-baseline.json`.
3. Run `run_demo`.
4. Confirm the transcript file is written to `outputs/demo-transcript.jsonl`.
5. Confirm speech output is available or cleanly falls back to mock mode.

## Live Demo Checks

- webcam permissions granted
- model file path exists
- room lighting is consistent
- hands stay inside the frame for several consecutive frames
- `confidence_threshold` and `hold_frames` are tuned for the room

## Backup Flow

If live capture becomes unstable:

1. rerun `run_demo --no-camera`
2. explain that the pipeline is falling back to the synthetic frame source
3. show the transcript log and trained model metrics
4. walk through the architecture and phase progress

This gives the team a credible fallback without pretending the live path is healthy when it is not.
