from __future__ import annotations

import unittest

from bridgelink_asl.config import AppConfig
from bridgelink_asl.speech import select_speech_adapter


class SpeechSelectionTests(unittest.TestCase):
    def test_default_mock_speech_selection(self) -> None:
        selection = select_speech_adapter(AppConfig())
        self.assertEqual(selection.resolved_provider, "mock")
        self.assertEqual(selection.adapter.name, "mock")

    def test_elevenlabs_without_credentials_falls_back(self) -> None:
        selection = select_speech_adapter(AppConfig(tts_provider="elevenlabs"))
        self.assertEqual(selection.resolved_provider, "mock")
        self.assertIn("requires", selection.fallback_reason.lower())
