from __future__ import annotations

import unittest
from pathlib import Path

from bridgelink_asl.dataset import DatasetRecord, load_dataset, validate_dataset


class DatasetTests(unittest.TestCase):
    def test_sample_dataset_is_valid(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "sample_landmarks.jsonl"
        records = load_dataset(dataset_path)
        self.assertEqual(validate_dataset(records), [])

    def test_validation_catches_inconsistent_lengths(self) -> None:
        records = [
            DatasetRecord(label="HELLO", split="train", landmarks=(0.1, 0.2)),
            DatasetRecord(label="YES", split="test", landmarks=(0.1,)),
        ]
        issues = validate_dataset(records, allowed_labels=("HELLO", "YES"))
        self.assertTrue(any("inconsistent" in issue.lower() for issue in issues))
