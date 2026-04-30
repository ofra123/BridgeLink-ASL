from __future__ import annotations

import re
import unicodedata


def normalize_label(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text.strip()
