from __future__ import annotations

import unittest

from bridgelink_asl.smoothing import PredictionSmoother
from bridgelink_asl.asl_types import Prediction


class SmoothingTests(unittest.TestCase):
    def test_prediction_requires_hold_frames(self) -> None:
        smoother = PredictionSmoother(hold_frames=3, confidence_threshold=0.5)

        self.assertIsNone(smoother.observe(Prediction(label="HELLO", confidence=0.9, frame_index=0)))
        self.assertIsNone(smoother.observe(Prediction(label="HELLO", confidence=0.9, frame_index=1)))
        emitted = smoother.observe(Prediction(label="HELLO", confidence=0.9, frame_index=2))

        self.assertIsNotNone(emitted)
        self.assertEqual(emitted.label, "HELLO")
        self.assertIsNone(smoother.observe(Prediction(label="HELLO", confidence=0.9, frame_index=3)))
