# Phase 2: Sentence-Window Dataset And Gesture Trace Pipeline

## Goal

Move from isolated picture/word recognition to short gesture windows. A sentence cannot be understood from one still image, so this phase creates clip-level data and a gesture trace that a VLM can use safely.

## Deliverables

- choose one external sentence-level ASL dataset for reference, preferably How2Sign for continuous ASL clips with English translations
- create a small team-owned demo dataset of 8-12 sentence clips using the project vocabulary
- define a JSONL clip schema with `clip_id`, `split`, `gloss`, `english`, frame references, optional landmarks, and notes
- add a sentence-window collector that groups frames over a configurable time span instead of treating each frame independently
- keep the existing word classifier output as a token trace: labels, confidence, timestamps, and frame ranges
- document dataset licensing and avoid committing large videos directly to Git

## Exit Criteria

- the repo contains a small metadata-only sample sentence dataset
- the team can collect or reference clip windows without changing the existing word-level fallback
- the sentence-window format is stable enough for tests and VLM prompting
- at least 5 demo sentences have known expected English outputs
- the team can explain which external dataset is being used and why
