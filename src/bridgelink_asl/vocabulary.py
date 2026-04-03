"""V1 sign vocabulary and seed landmark templates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignDefinition:
    label: str
    text: str
    description: str
    seed_landmarks: tuple[float, ...]


V1_SIGNS: tuple[SignDefinition, ...] = (
    SignDefinition("HELLO", "Hello", "Greeting sign for starting an interaction.", (0.12, 0.18, 0.81, 0.75, 0.22, 0.68)),
    SignDefinition("YES", "Yes", "Positive confirmation.", (0.82, 0.76, 0.14, 0.19, 0.64, 0.28)),
    SignDefinition("NO", "No", "Negative response.", (0.74, 0.22, 0.58, 0.31, 0.14, 0.82)),
    SignDefinition("THANK_YOU", "Thank you", "Polite appreciation phrase.", (0.21, 0.84, 0.63, 0.16, 0.52, 0.48)),
    SignDefinition("PLEASE", "Please", "Polite request phrase.", (0.36, 0.71, 0.27, 0.62, 0.83, 0.22)),
    SignDefinition("HELP", "Help", "Call for assistance.", (0.61, 0.14, 0.82, 0.44, 0.25, 0.56)),
    SignDefinition("STOP", "Stop", "Instruction to pause or halt.", (0.91, 0.44, 0.24, 0.87, 0.18, 0.34)),
    SignDefinition("EAT", "Eat", "Food-related request or statement.", (0.33, 0.27, 0.73, 0.58, 0.48, 0.14)),
    SignDefinition("DRINK", "Drink", "Drink-related request or statement.", (0.44, 0.88, 0.21, 0.49, 0.66, 0.17)),
    SignDefinition("WANT", "Want", "Desire or preference signal.", (0.69, 0.34, 0.41, 0.78, 0.57, 0.24)),
    SignDefinition("MORE", "More", "Request for more of something.", (0.27, 0.55, 0.91, 0.29, 0.36, 0.73)),
    SignDefinition("FINISHED", "Finished", "Completion or done signal.", (0.58, 0.63, 0.32, 0.24, 0.88, 0.41)),
)

SIGN_BY_LABEL = {sign.label: sign for sign in V1_SIGNS}
LANDMARK_FEATURE_LENGTH = len(V1_SIGNS[0].seed_landmarks)
DEFAULT_DEMO_SEQUENCE = tuple(sign.label for sign in V1_SIGNS[:6])


def get_v1_labels() -> tuple[str, ...]:
    """Return the supported V1 label set."""

    return tuple(SIGN_BY_LABEL)


def get_seed_landmarks(label: str) -> tuple[float, ...]:
    """Return the built-in seed landmarks for a known label."""

    return SIGN_BY_LABEL[label].seed_landmarks
