"""Sign label to text translation helpers."""

from __future__ import annotations

from .types import Prediction, TranslationEvent
from .vocabulary import SIGN_BY_LABEL


class SignTranslator:
    """Translate classifier labels into user-facing text."""

    def to_text(self, label: str) -> str:
        definition = SIGN_BY_LABEL.get(label)
        if definition:
            return definition.text
        return label.replace("_", " ").title()

    def to_event(self, prediction: Prediction, tts_provider: str) -> TranslationEvent:
        return TranslationEvent(
            label=prediction.label,
            text=self.to_text(prediction.label),
            confidence=prediction.confidence,
            frame_index=prediction.frame_index,
            tts_provider=tts_provider,
        )
