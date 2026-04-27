from __future__ import annotations

from bridgelink_asl.sentence_labels import normalize_manifest_rows, normalize_sentence_label


def test_normalize_sentence_label_merges_known_duplicate_classes() -> None:
    assert normalize_sentence_label("Hi!") == "Hi."
    assert normalize_sentence_label("Hi.") == "Hi."
    assert normalize_sentence_label("O.k.") == "Okay."
    assert normalize_sentence_label("Okay?") == "Okay."
    assert normalize_sentence_label("Alright.") == "All right."
    assert normalize_sentence_label("All right.") == "All right."
    assert normalize_sentence_label("Here we go.") == "Here we go."


def test_normalize_manifest_rows_updates_sentence_and_english_fields() -> None:
    rows = [
        {"clip_id": "a", "sentence_label": "Hi!", "english": "Hi!"},
        {"clip_id": "b", "sentence_label": "Hi.", "english": "Hi."},
        {"clip_id": "c", "sentence_label": "Okay?", "english": "Okay?"},
        {"clip_id": "d", "sentence_label": "There we go.", "english": "There we go."},
    ]

    normalized, summary = normalize_manifest_rows(rows)

    assert [row["sentence_label"] for row in normalized] == ["Hi.", "Hi.", "Okay.", "There we go."]
    assert normalized[0]["english"] == "Hi."
    assert normalized[2]["english"] == "Okay."
    assert summary.changed_rows == 2
    assert summary.classes_before == 4
    assert summary.classes_after == 3
    assert summary.merged_groups["Hi."] == ["Hi!", "Hi."]
