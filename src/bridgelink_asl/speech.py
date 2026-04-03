"""Speech adapter selection with safe fallbacks."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import AppConfig


@dataclass
class MockSpeechAdapter:
    """Fallback speech adapter that records utterances without audio output."""

    name: str = "mock"
    history: list[str] = field(default_factory=list)

    def speak(self, text: str) -> None:
        self.history.append(text)
        print(f"[TTS:{self.name}] {text}")


@dataclass
class Pyttsx3SpeechAdapter:
    """Optional local text-to-speech adapter."""

    name: str = "pyttsx3"

    def __post_init__(self) -> None:
        import pyttsx3  # type: ignore  # pragma: no cover - optional dependency

        self._engine = pyttsx3.init()

    def speak(self, text: str) -> None:  # pragma: no cover - optional dependency
        self._engine.say(text)
        self._engine.runAndWait()


@dataclass
class ElevenLabsSpeechAdapter:
    """Optional ElevenLabs API adapter."""

    api_key: str
    voice_id: str
    name: str = "elevenlabs"

    def speak(self, text: str) -> None:  # pragma: no cover - networked optional dependency
        payload = json.dumps(
            {
                "text": text,
                "model_id": "eleven_multilingual_v2",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url=f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
            data=payload,
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15):
                return
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ElevenLabs request failed: {exc}") from exc


@dataclass
class SpeechAdapterSelection:
    """The requested provider, the resolved adapter, and any fallback reason."""

    adapter: object
    requested_provider: str
    resolved_provider: str
    fallback_reason: str | None = None


def select_speech_adapter(config: AppConfig) -> SpeechAdapterSelection:
    """Resolve the configured speech provider to a usable adapter."""

    requested = config.tts_provider.lower()
    if requested in {"mock", "console"}:
        return SpeechAdapterSelection(
            adapter=MockSpeechAdapter(name=requested),
            requested_provider=requested,
            resolved_provider=requested,
        )

    if requested == "pyttsx3":
        try:
            adapter = Pyttsx3SpeechAdapter()
            return SpeechAdapterSelection(
                adapter=adapter,
                requested_provider=requested,
                resolved_provider=adapter.name,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            return SpeechAdapterSelection(
                adapter=MockSpeechAdapter(),
                requested_provider=requested,
                resolved_provider="mock",
                fallback_reason=f"Fell back to mock speech because pyttsx3 is unavailable: {exc}",
            )

    if requested == "elevenlabs":
        if not config.elevenlabs_api_key or not config.elevenlabs_voice_id:
            return SpeechAdapterSelection(
                adapter=MockSpeechAdapter(),
                requested_provider=requested,
                resolved_provider="mock",
                fallback_reason="ElevenLabs requires both ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID.",
            )
        return SpeechAdapterSelection(
            adapter=ElevenLabsSpeechAdapter(
                api_key=config.elevenlabs_api_key,
                voice_id=config.elevenlabs_voice_id,
            ),
            requested_provider=requested,
            resolved_provider=requested,
        )

    return SpeechAdapterSelection(
        adapter=MockSpeechAdapter(),
        requested_provider=requested,
        resolved_provider="mock",
        fallback_reason=f"Unknown speech provider '{requested}', using mock instead.",
    )
