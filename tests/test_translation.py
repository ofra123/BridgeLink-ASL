from __future__ import annotations

import unittest

from bridgelink_asl.translation import SignTranslator


class TranslationTests(unittest.TestCase):
    def test_known_label_translation(self) -> None:
        translator = SignTranslator()
        self.assertEqual(translator.to_text("THANK_YOU"), "Thank you")

    def test_unknown_label_translation(self) -> None:
        translator = SignTranslator()
        self.assertEqual(translator.to_text("custom_sign"), "Custom Sign")
