"""CLI for the BridgeLink ASL starter demo."""

from __future__ import annotations

import argparse

from ..config import load_config
from ..pipeline import build_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BridgeLink ASL demo loop.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--frames", type=int, help="Number of frames to process.")
    parser.add_argument("--model-path", help="Path to a saved model file.")
    parser.add_argument("--tts-provider", help="Speech provider to use.")
    parser.add_argument("--camera-index", type=int, help="Camera index for OpenCV.")
    parser.add_argument("--transcript-path", help="Where to save transcript JSONL events.")
    parser.add_argument("--no-camera", action="store_true", help="Force the synthetic frame source.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overrides = {
        "max_frames": args.frames,
        "model_path": args.model_path,
        "tts_provider": args.tts_provider,
        "camera_index": args.camera_index,
        "transcript_path": args.transcript_path,
    }
    if args.no_camera:
        overrides["use_camera"] = False

    config = load_config(args.config, overrides=overrides)
    session = build_session(config)
    summary = session.run()

    print("BridgeLink ASL demo complete.")
    print(f"Frames processed: {summary.frames_processed}")
    print(f"Frame source: {summary.frame_source}")
    print(f"Classifier source: {summary.classifier_source}")
    print(f"TTS provider: {summary.tts_provider}")
    if session.speech_selection.fallback_reason:
        print(session.speech_selection.fallback_reason)

    if summary.events:
        print("Stable translation events:")
        for event in summary.events:
            print(f"- frame {event.frame_index}: {event.text} ({event.confidence:.2f})")
    else:
        print("No stable translation events were emitted.")

    print(f"Transcript path: {config.transcript_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
