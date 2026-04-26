from __future__ import annotations

import unittest
from pathlib import Path

from bridgelink_asl.clip_dataset import load_clip_dataset
from bridgelink_asl.metrics import compute_classification_metrics
from bridgelink_asl.project_assets import (
    build_comparison_results,
    build_dataset_summary,
    report_summary_markdown,
    write_comparison_artifacts,
)


class MetricsAndProjectAssetsTests(unittest.TestCase):
    def test_classification_metrics_include_macro_scores(self) -> None:
        metrics = compute_classification_metrics(
            expected=["A", "A", "B", "B"],
            predicted=["A", "B", "B", "B"],
        )

        self.assertEqual(metrics.total, 4)
        self.assertEqual(metrics.correct, 3)
        self.assertEqual(metrics.accuracy, 0.75)
        self.assertIn("A", metrics.per_label)
        self.assertIn("B", metrics.confusion_matrix)

    def test_wlasl_hybrid_manifest_builds_assets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "data" / "vlm_eval_wlasl25_cnn" / "wlasl25_cnn_hybrid_eval.jsonl"
        records = load_clip_dataset(manifest)

        summary = build_dataset_summary(records)
        comparison = build_comparison_results(records)
        report = report_summary_markdown()

        self.assertEqual(summary["dataset"], "WLASL-25 hybrid evaluation subset")
        self.assertEqual(summary["total_clips"], 36)
        self.assertIn("cnn_metrics", comparison)
        self.assertIn("<svg", comparison["comparison_chart_svg"])
        self.assertIn("CVPR-Style", report)

    def test_write_comparison_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "data" / "vlm_eval_wlasl25_cnn" / "wlasl25_cnn_hybrid_eval.jsonl"
        output_dir = root / ".tmp-tests" / "report-assets"

        paths = write_comparison_artifacts(manifest, output_dir)

        self.assertTrue(paths["dataset_summary"].exists())
        self.assertTrue(paths["comparison_results"].exists())
        self.assertTrue(paths["model_comparison"].read_text(encoding="utf-8").startswith("<svg"))
