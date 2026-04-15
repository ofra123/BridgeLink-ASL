from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from bridgelink_asl.space_inference import analyze_video, result_to_markdown, run_space_poc


class SpaceInferenceTests(unittest.TestCase):
    def test_space_poc_runs_without_video(self) -> None:
        result = run_space_poc(None, "Compare", vlm_model_id="Qwen/Qwen2.5-VL-32B-Instruct-AWQ")

        self.assertEqual(result["status"], "proof_of_concept")
        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["final_sentence"], "Please help.")
        self.assertEqual(result["cnn"]["status"], "poc_video_feature_baseline")
        self.assertEqual(result["vlm"]["status"], "grounded_mock_vlm")
        self.assertFalse(result["video_features"]["exists"])

    def test_space_poc_can_use_file_metadata_fallback(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"space-{uuid.uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        video_path = scratch_dir / "clip.mp4"
        video_path.write_bytes(b"not-a-real-video-but-valid-file-for-fallback")

        summary = analyze_video(video_path)
        result = run_space_poc(str(video_path), "CNN", vlm_model_id="unused")

        self.assertTrue(summary.exists)
        self.assertGreater(summary.file_size_bytes, 0)
        self.assertEqual(result["cnn"]["status"], "poc_video_feature_baseline")
        self.assertEqual(result["vlm"]["status"], "not_run")

    def test_space_result_renders_markdown(self) -> None:
        result = run_space_poc(None, "VLM", vlm_model_id="Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
        markdown = result_to_markdown(result)

        self.assertIn("Final Sentence", markdown)
        self.assertIn("Please help.", markdown)
