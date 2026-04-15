# Scope And Success Criteria

## Current Supported Vocabulary

The current word-level target set remains intentionally small and demo-friendly:

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

## Current Baseline Success Criteria

The current baseline is successful when:

- a teammate can clone the repo and install it locally
- `run_demo` produces at least one translated event and transcript entry
- `train_model` saves a baseline artifact from the sample dataset
- `evaluate_model` produces metrics without manual editing
- the output path stays resilient even if ElevenLabs, VLM access, or webcam access is unavailable

## Next Scope: Sentence Mode

The next project target is not full open-ended ASL translation. It is controlled sentence-level interpretation over short gesture windows.

Sentence mode should support:

- short 2-5 second clips
- sampled keyframes from the clip
- a token trace from the existing word-level classifier
- VLM interpretation into one English sentence
- mock/offline fallback when the VLM is unavailable

## Still Out Of Scope

- open-ended full ASL translation
- long continuous conversations
- production accessibility claims
- production deployment
- mandatory Firebase integration
- mandatory cloud TTS during local demos
- mandatory cloud VLM during local tests
