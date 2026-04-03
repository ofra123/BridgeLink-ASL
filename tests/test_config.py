from __future__ import annotations

import json
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from bridgelink_asl.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_merges_file_and_env(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"config-{uuid.uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        config_path = scratch_dir / "demo.json"
        config_path.write_text(
            json.dumps(
                {
                    "camera_index": 2,
                    "model_path": "models/custom.json",
                    "demo_sequence": ["hello", "help"],
                    "use_camera": False,
                }
            ),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"BRIDGELINK_TTS_PROVIDER": "console"}, clear=False):
            config = load_config(config_path)

        self.assertEqual(config.camera_index, 2)
        self.assertEqual(config.tts_provider, "console")
        self.assertEqual(config.demo_sequence, ("HELLO", "HELP"))
        self.assertFalse(config.use_camera)
        self.assertTrue(str(config.model_path).endswith(str(Path("models") / "custom.json")))
