from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from bridgelink_asl.evaluation import evaluate_saved_model, save_metrics
from bridgelink_asl.training import train_centroid_model


class TrainingAndEvaluationTests(unittest.TestCase):
    def test_training_and_evaluation_round_trip(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "sample_landmarks.jsonl"

        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"training-{uuid.uuid4().hex}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        model_path = scratch_dir / "baseline.json"
        metrics_path = scratch_dir / "metrics.json"

        summary = train_centroid_model(dataset_path, model_path)
        metrics = evaluate_saved_model(model_path, dataset_path, split="test")
        save_metrics(metrics, metrics_path)

        self.assertTrue(model_path.exists())
        self.assertEqual(summary.records_used, 12)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertTrue(metrics_path.exists())
