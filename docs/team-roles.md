# Team Roles And Next Work

## Role Split

Trey owns data prep. Trey should get How2Sign access, create the clip manifest, define train/val/test splits, extract sampled frames, and keep raw videos outside Git.

Dalen owns the wrapper. Dalen should make one command or app mode that can run `cnn`, `vlm`, and `compare` over the same clip manifest, then log predictions, latency, and failures.

Omar owns the CNN. Omar should train the sampled-frame CNN baseline from the manifest, save model artifacts locally, and export metrics that can be compared against the VLM.

Frank owns the VLM. Frank should set up Qwen2.5-VL locally, build the strict JSON prompt, and produce sentence predictions using the same clip records as the CNN.

## Immediate Next Tasks

Trey:

- Download or request How2Sign access.
- Select a small subset of sentence clips that match our V1 vocabulary as closely as possible.
- Create `data/processed/how2sign_clips.jsonl` using the sample schema.
- Extract 16 sampled frames per clip into local `data/interim/frames/...`.
- Do not commit raw videos or large frame dumps.

Dalen:

- Add wrapper modes for `cnn`, `vlm`, and `compare`.
- Make sure both model paths consume the same clip manifest.
- Write results to `outputs/comparison-results.jsonl`.
- Track latency and failures for each clip.

Omar:

- Start with `train_cnn_model --clips data/processed/sample_sentence_clips.jsonl --dry-run`.
- Once Trey has frame paths, install `pip install -e ".[training]"`.
- Train `models/cnn-baseline.keras` from the How2Sign/team clip manifest.
- Report held-out test accuracy, confusion examples, and what signs/sentences fail most often.

Frank:

- Download or prepare `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`, with 7B fallback if hardware is tight and 72B as a stretch if strong hardware is available.
- Build a `SentenceInterpreter` interface with mock and local Qwen providers.
- Prompt the VLM to return strict JSON with gloss, sentence, confidence, and failure reason.
- Compare VLM output against the same `english` targets in the manifest.

## Definition Of Done For The Comparison

- Both CNN and VLM run against the same test clips.
- Results include accuracy or rubric score, latency, and failure notes.
- The final demo can show at least one CNN prediction and one VLM sentence prediction.
- The team can explain why CNN is the controlled baseline and why VLM is better suited to sentence-level understanding.
