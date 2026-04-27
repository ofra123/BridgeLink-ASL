from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

from bridgelink_asl.clip_dataset import load_clip_dataset, validate_clip_dataset
from bridgelink_asl.cnn import (
    CnnModelConfig,
    build_cnn_training_plan,
    describe_cnn_baseline,
    select_frame_paths,
)


class ClipDatasetAndCnnTests(unittest.TestCase):
    def test_loads_clip_manifest_and_builds_cnn_plan(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"clips-{uuid.uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = scratch_dir / "clips.jsonl"
        manifest_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "clip_id": "team_hello_want_drink_001",
                            "split": "train",
                            "source": "team",
                            "gloss": ["hello", "want", "drink"],
                            "english": "Hello, I want a drink.",
                            "sampled_frames": ["frames/001.jpg", "frames/002.jpg"],
                        }
                    ),
                    json.dumps(
                        {
                            "clip_id": "team_no_stop_001",
                            "split": "test",
                            "source": "team",
                            "gloss": ["NO", "STOP"],
                            "english": "No, stop.",
                            "sampled_frames": ["frames/003.jpg"],
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        records = load_clip_dataset(manifest_path)
        issues = validate_clip_dataset(records, require_sampled_frames=True)
        plan = build_cnn_training_plan(records, CnnModelConfig(frame_count=8, image_size=96))

        self.assertEqual(issues, [])
        self.assertEqual(records[0].label, "HELLO WANT DRINK")
        self.assertEqual(plan.input_shape, (8, 96, 96, 3))
        self.assertEqual(plan.train_records, 1)
        self.assertEqual(plan.test_records, 1)
        self.assertEqual(plan.labels, ("HELLO WANT DRINK", "NO STOP"))

    def test_clip_validation_reports_duplicates_and_missing_fields(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"bad-clips-{uuid.uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = scratch_dir / "bad-clips.jsonl"
        manifest_path.write_text(
            "\n".join(
                [
                    json.dumps({"clip_id": "duplicate", "split": "train", "gloss": [], "english": ""}),
                    json.dumps({"clip_id": "duplicate", "split": "holdout", "gloss": ["HELP"], "english": "Help."}),
                ]
            ),
            encoding="utf-8",
        )

        issues = validate_clip_dataset(load_clip_dataset(manifest_path), require_sampled_frames=True)

        self.assertTrue(any("duplicate clip IDs" in issue for issue in issues))
        self.assertTrue(any("unsupported splits" in issue for issue in issues))
        self.assertTrue(any("without gloss labels" in issue for issue in issues))
        self.assertTrue(any("without English targets" in issue for issue in issues))
        self.assertTrue(any("without sampled frame paths" in issue for issue in issues))

    def test_loads_project_root_relative_sampled_frames(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"root-relative-{uuid.uuid4().hex}"
        manifest_dir = scratch_dir / "data" / "processed"
        frame_path = scratch_dir / "data" / "interim" / "frames" / "001.jpg"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"jpg")

        manifest_path = manifest_dir / "clips.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "clip_id": "root_relative_clip",
                    "split": "train",
                    "source": "team",
                    "gloss": ["HELLO"],
                    "english": "Hello.",
                    "sampled_frames": ["data/interim/frames/001.jpg"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        records = load_clip_dataset(manifest_path)

        self.assertEqual(records[0].sampled_frames, (frame_path.resolve(),))
        self.assertEqual(validate_clip_dataset(records, require_sampled_frames=True, require_files=True), [])

    def test_select_frame_paths_pads_and_uniformly_samples(self) -> None:
        paths = tuple(Path(f"frame_{index}.jpg") for index in range(5))

        self.assertEqual(select_frame_paths(paths[:2], 4), (paths[0], paths[1], paths[1], paths[1]))
        self.assertEqual(select_frame_paths(paths, 3), (paths[0], paths[2], paths[4]))

    def test_cnn_description_is_available_without_tensorflow(self) -> None:
        config = CnnModelConfig(frame_count=12, image_size=128)
        description = describe_cnn_baseline(config, num_classes=4)

        self.assertEqual(description["model_family"], "sentence-3d-cnn")
        self.assertEqual(description["input_shape"], [12, 128, 128, 3])
        self.assertEqual(description["num_classes"], 4)
        self.assertEqual(description["temporal_strategy"], "Conv3D over sampled RGB clip volumes")

    def test_prepare_clip_dataset_writes_absolute_sampled_frame_paths(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"prepare-clips-{uuid.uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = scratch_dir / "data" / "processed" / "clips.jsonl"
        output_manifest = scratch_dir / "data" / "processed" / "clips.frames.jsonl"
        frame_root = scratch_dir / "data" / "interim" / "frames"
        video_path = scratch_dir / "clips" / "clip.mp4"
        frame_path = frame_root / "clip-001" / "frame_000.jpg"

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.parent.mkdir(parents=True, exist_ok=True)

        video_path.write_bytes(b"not-a-real-video")
        frame_path.write_bytes(b"jpg")
        manifest_path.write_text(json.dumps({"clip_id": "clip-001", "video_path": str(video_path)}) + "\n", encoding="utf-8")

        module_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_clip_dataset.py"
        spec = importlib.util.spec_from_file_location("prepare_clip_dataset_test", module_path)
        if spec is None or spec.loader is None:
            self.fail("Could not load prepare_clip_dataset.py for testing.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with mock.patch.dict(sys.modules, {"cv2": object()}):
            with mock.patch.object(module, "_sample_video_frames", return_value=[frame_path]):
                module._extract_frames_to_manifest(manifest_path, output_manifest, frame_root, 1)

        payload = json.loads(output_manifest.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(payload["sampled_frames"], [frame_path.resolve().as_posix()])
