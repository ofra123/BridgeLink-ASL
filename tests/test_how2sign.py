from __future__ import annotations

import csv
import unittest
import uuid
from pathlib import Path

from bridgelink_asl.how2sign import build_how2sign_sentence_manifest


class How2SignManifestTests(unittest.TestCase):
    def test_builds_closed_vocabulary_manifest_from_repeated_sentences(self) -> None:
        tmp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        scratch_dir = tmp_root / f"how2sign-{uuid.uuid4().hex}"
        translation_dir = scratch_dir / "translations"
        clip_dir = scratch_dir / "clips"
        translation_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "VIDEO_ID": "1",
                "VIDEO_NAME": "train_a",
                "SENTENCE_ID": "10",
                "SENTENCE_NAME": "train_a",
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "1.0",
                "SENTENCE": "Good.",
            },
            {
                "VIDEO_ID": "2",
                "VIDEO_NAME": "train_b",
                "SENTENCE_ID": "11",
                "SENTENCE_NAME": "train_b",
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "1.0",
                "SENTENCE": "Good.",
            },
            {
                "VIDEO_ID": "3",
                "VIDEO_NAME": "val_a",
                "SENTENCE_ID": "12",
                "SENTENCE_NAME": "val_a",
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "1.0",
                "SENTENCE": "Hi!",
            },
            {
                "VIDEO_ID": "4",
                "VIDEO_NAME": "test_a",
                "SENTENCE_ID": "13",
                "SENTENCE_NAME": "test_a",
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "1.0",
                "SENTENCE": "Hi!",
            },
            {
                "VIDEO_ID": "5",
                "VIDEO_NAME": "missing_clip",
                "SENTENCE_ID": "14",
                "SENTENCE_NAME": "missing_clip",
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "1.0",
                "SENTENCE": "Good.",
            },
        ]

        self._write_csv(translation_dir / "how2sign_realigned_train.csv", rows[:2])
        self._write_csv(translation_dir / "how2sign_realigned_val.csv", rows[2:3])
        self._write_csv(translation_dir / "how2sign_realigned_test.csv", rows[3:])

        for clip_name in ("train_a", "train_b", "val_a", "test_a"):
            (clip_dir / f"{clip_name}.mp4").write_bytes(b"demo")

        manifest_rows, summary = build_how2sign_sentence_manifest(
            translation_dir=translation_dir,
            clip_dir=clip_dir,
            min_count=2,
            max_classes=2,
            max_samples_per_class=3,
        )

        self.assertEqual(summary["matched_clip_rows"], 4)
        self.assertEqual(summary["selected_rows"], 4)
        self.assertEqual(summary["selected_classes"], 2)
        self.assertEqual(summary["original_split_counts"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(summary["split_counts"], {"train": 2, "val": 2})
        self.assertEqual(summary["split_strategy"], "deterministic_per_class_70_15_15")
        self.assertEqual({row.sentence_label for row in manifest_rows}, {"Good.", "Hi!"})
        self.assertEqual(manifest_rows[0].gloss, ("GOOD",))
        self.assertTrue(all(row.video_path.exists() for row in manifest_rows))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        fieldnames = [
            "VIDEO_ID",
            "VIDEO_NAME",
            "SENTENCE_ID",
            "SENTENCE_NAME",
            "START_REALIGNED",
            "END_REALIGNED",
            "SENTENCE",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
