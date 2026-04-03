# V1 Scope And Success Criteria

## Supported Vocabulary

The phase-1 and phase-2 target set is intentionally small and demo-friendly:

- `HELLO`
- `YES`
- `NO`
- `THANK_YOU`
- `PLEASE`
- `HELP`
- `STOP`
- `EAT`
- `DRINK`
- `WANT`
- `MORE`
- `FINISHED`

## Success Criteria

The team can treat the starter milestone as complete when:

- a teammate can clone the repo and install it locally
- `run_demo` produces at least one translated event and transcript entry
- `train_model` saves a baseline artifact from the sample dataset
- `evaluate_model` produces metrics without manual editing
- the output path stays resilient even if ElevenLabs or webcam access is unavailable

## Out Of Scope For V1

- full ASL sentence translation
- continuous signing with grammar modeling
- production deployment
- mandatory Firebase integration
- mandatory cloud TTS during local demos
