"""Validate a clip manifest and optionally extract sampled video frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgelink_asl.clip_dataset import load_clip_dataset, validate_clip_dataset  # noqa: E402
from bridgelink_asl.project_assets import build_dataset_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate a BridgeLink clip manifest.")
    parser.add_argument("--manifest", default="data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl")
    parser.add_argument("--summary-output", default="results/dataset-summary.json")
    parser.add_argument("--require-frames", action="store_true")
    parser.add_argument("--extract-frames", action="store_true", help="Extract sampled frames from local video paths.")
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--frame-root", default="data/interim/frames")
    parser.add_argument("--output-manifest", help="Optional manifest path with sampled frame paths added.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if args.extract_frames:
        output_manifest = Path(args.output_manifest or args.manifest)
        _extract_frames_to_manifest(manifest_path, output_manifest, Path(args.frame_root), args.frame_count)
        manifest_path = output_manifest

    records = load_clip_dataset(manifest_path)
    issues = validate_clip_dataset(records, require_sampled_frames=args.require_frames)
    summary = build_dataset_summary(records)
    summary["validation_issues"] = issues

    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if issues else 0


def _extract_frames_to_manifest(
    manifest_path: Path,
    output_manifest: Path,
    frame_root: Path,
    frame_count: int,
) -> None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame extraction. Install opencv-python-headless.") from exc

    updated_rows: list[dict[str, object]] = []
    for payload in _read_jsonl(manifest_path):
        clip_id = str(payload["clip_id"])
        video_path = Path(str(payload.get("video_path", "")))
        if video_path.exists():
            clip_frame_dir = frame_root / clip_id
            sampled = _sample_video_frames(cv2, video_path, clip_frame_dir, frame_count)
            payload["sampled_frames"] = [str(path).replace("\\", "/") for path in sampled]
            payload["notes"] = str(payload.get("notes", "")).strip() + " Sampled frames extracted."
        else:
            payload["notes"] = str(payload.get("notes", "")).strip() + " Video file missing during extraction."
        updated_rows.append(payload)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text("\n".join(json.dumps(row) for row in updated_rows) + "\n", encoding="utf-8")


def _sample_video_frames(cv2, video_path: Path, output_dir: Path, frame_count: int) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indexes = _uniform_indexes(total_frames, frame_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for output_index, frame_index in enumerate(indexes):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        output_path = output_dir / f"frame_{output_index:03d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved.append(output_path)
    capture.release()
    return saved


def _uniform_indexes(total_frames: int, frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if total_frames <= 1:
        return [0] * frame_count
    if frame_count == 1:
        return [0]
    last = total_frames - 1
    return [round(index * last / (frame_count - 1)) for index in range(frame_count)]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
