"""Data collection script for generating landmark JSONL datasets using MediaPipe."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import mediapipe as mp


print("RUNNING DATA COLLECTION SCRIPT")

OUTPUT_PATH = Path("data/processed/asl_dataset_v1.jsonl")
LABELS = [
    "HELLO", "YES", "NO", "THANK_YOU", "PLEASE",
    "HELP", "STOP", "EAT", "DRINK", "WANT",
    "MORE", "FINISHED",
]


def assign_split() -> str:
    """Randomly assign dataset split."""
    r = random.random()
    if r < 0.7:
        return "train"
    elif r < 0.9:
        return "val"
    return "test"


def extract_landmarks(hand_landmarks) -> list[float]:
    """Flatten MediaPipe landmarks into a list of floats."""
    values: list[float] = []
    for lm in hand_landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])
    return values


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(1)
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Starting data collection...")
    print("Press keys 0-9 to label signs. Press 'x' to quit.")
    print("Label mapping:")
    for i, label in enumerate(LABELS):
        print(f"{i}: {label}")

    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(image_rgb)

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("x"):
                break

            if results.multi_hand_landmarks:
                label_index = None

                if ord("0") <= key <= ord("9"):
                    label_index = key - ord("0")

                elif key == ord("q"):
                    label_index = 10
                elif key == ord("w"):
                    label_index = 11

                if label_index is not None and label_index < len(LABELS):
                    label = LABELS[label_index]

                    hand_landmarks = results.multi_hand_landmarks[0]
                    values = extract_landmarks(hand_landmarks)

                    record = {
                        "label": label,
                        "split": assign_split(),
                        "landmarks": values,
                    }

                    f.write(json.dumps(record) + "\n")
                    print(f"Saved sample for {label}")

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
