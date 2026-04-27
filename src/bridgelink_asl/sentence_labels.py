"""Sentence-label normalization helpers for closed-vocabulary How2Sign CNN runs."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


_WHITESPACE_RE = re.compile(r"\s+")
_TERMINAL_PUNCT_RE = re.compile(r"[.!?]+$")


@dataclass(frozen=True)
class SentenceNormalizationSummary:
    """Human-readable summary of manifest label normalization."""

    total_rows: int
    changed_rows: int
    classes_before: int
    classes_after: int
    label_counts_before: dict[str, int]
    label_counts_after: dict[str, int]
    merged_groups: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "changed_rows": self.changed_rows,
            "classes_before": self.classes_before,
            "classes_after": self.classes_after,
            "label_counts_before": self.label_counts_before,
            "label_counts_after": self.label_counts_after,
            "merged_groups": self.merged_groups,
        }


def normalize_sentence_label(label: str, strategy: str = "conservative") -> str:
    """Return a canonical label for a repeated-sentence CNN class."""

    text = _WHITESPACE_RE.sub(" ", label.strip())
    if not text:
        return text

    key = _canonical_key(text)
    if strategy == "conservative":
        return _normalize_conservative(text, key)
    raise ValueError(f"Unsupported sentence normalization strategy: {strategy}")


def normalize_manifest_rows(
    rows: Iterable[dict[str, Any]],
    *,
    strategy: str = "conservative",
) -> tuple[list[dict[str, Any]], SentenceNormalizationSummary]:
    """Normalize sentence labels inside JSONL manifest rows."""

    materialized = [dict(row) for row in rows]
    counts_before = Counter(_row_label(row) for row in materialized if _row_label(row))
    normalized_rows: list[dict[str, Any]] = []
    changed_rows = 0
    merged_groups: dict[str, set[str]] = defaultdict(set)

    for row in materialized:
        original_label = _row_label(row)
        if not original_label:
            normalized_rows.append(row)
            continue

        canonical_label = normalize_sentence_label(original_label, strategy=strategy)
        if canonical_label != original_label:
            changed_rows += 1
            row["original_sentence_label"] = original_label
            row["sentence_label"] = canonical_label
            if str(row.get("english", "")).strip() == original_label:
                row["english"] = canonical_label
            merged_groups[canonical_label].add(original_label)
            merged_groups[canonical_label].add(canonical_label)
        normalized_rows.append(row)

    counts_after = Counter(_row_label(row) for row in normalized_rows if _row_label(row))
    summary = SentenceNormalizationSummary(
        total_rows=len(materialized),
        changed_rows=changed_rows,
        classes_before=len(counts_before),
        classes_after=len(counts_after),
        label_counts_before=dict(sorted(counts_before.items())),
        label_counts_after=dict(sorted(counts_after.items())),
        merged_groups={
            canonical: sorted(labels)
            for canonical, labels in sorted(merged_groups.items())
        },
    )
    return normalized_rows, summary


def _row_label(row: dict[str, Any]) -> str:
    return str(row.get("sentence_label") or "").strip()


def _canonical_key(label: str) -> str:
    text = _WHITESPACE_RE.sub(" ", label.strip().lower())
    text = _TERMINAL_PUNCT_RE.sub("", text)
    text = text.replace("o.k", "okay")
    text = text.replace("alright", "all right")
    return text


def _normalize_conservative(original: str, key: str) -> str:
    if key == "hi":
        return "Hi."
    if key == "okay":
        return "Okay."
    if key == "all right":
        return "All right."
    return original.strip()
