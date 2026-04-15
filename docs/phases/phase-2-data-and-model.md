# Phase 2: Sentence-Window Dataset And CNN Baseline

## Goal

Move from isolated picture/word recognition to short gesture windows and implement the professor-requested CNN baseline. A sentence cannot be understood from one still image, so this phase creates clip-level data that both CNN and VLM paths can share.

## Deliverables

- choose How2Sign as the external sentence-level ASL dataset for reference and experiments
- create a small team-owned demo dataset of 8-12 sentence clips using the project vocabulary
- define a JSONL clip schema with `clip_id`, `split`, `gloss`, `english`, frame references, optional landmarks, and notes
- add a sentence-window collector that groups frames over a configurable time span instead of treating each frame independently
- extract fixed-length sampled frame stacks for CNN training, starting with 16 frames per clip
- implement and train a sampled-frame CNN baseline that predicts the sentence/gloss class
- keep the existing word classifier output as an optional token trace for later VLM grounding
- document dataset licensing and avoid committing large videos directly to Git

## Implementation Notes

- Build the How2Sign subset manifest, splits, and frame extraction first.
- Then train the CNN architecture and save model artifact notes plus test accuracy.

## Exit Criteria

- the repo contains a small metadata-only sample sentence dataset
- the team can collect or reference clip windows without changing the existing word-level fallback
- the sentence-window format is stable enough for both CNN training and VLM prompting
- the CNN dry-run works on the manifest without TensorFlow
- CNN training works once sampled frames and TensorFlow dependencies are installed
- at least 5 demo sentences have known expected English outputs
- the team can explain which external dataset is being used and why
