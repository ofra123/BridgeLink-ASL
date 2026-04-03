from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from bridgelink_asl.config import AppConfig
from bridgelink_asl.pipeline import build_session


class PipelineTests(unittest.TestCase):
    def test_synthetic_demo_emits_events_and_transcript(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"pipeline-{uuid.uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = scratch_dir / "demo-transcript.jsonl"
        config = AppConfig(
            use_camera=False,
            hold_frames=2,
            confidence_threshold=0.7,
            max_frames=6,
            transcript_path=transcript_path,
            demo_sequence=("HELLO", "YES"),
        )

        session = build_session(config)
        summary = session.run()

        self.assertEqual(summary.frame_source, "synthetic")
        self.assertGreaterEqual(len(summary.events), 2)
        self.assertTrue(transcript_path.exists())
        self.assertEqual(len(session.speech_selection.adapter.history), len(summary.events))
