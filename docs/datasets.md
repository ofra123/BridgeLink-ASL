# Dataset Plan

## Short Answer

Use a two-layer dataset strategy:

- Primary sentence dataset: How2Sign
- Project-owned demo dataset: 8-12 short clips recorded by the team
- Optional isolated-sign support datasets: WLASL or ASL Citizen

Do not rely only on Sign Language MNIST for this project. It is useful for alphabet experiments, but it does not support sentence-level ASL translation.

## Recommended Primary Dataset: How2Sign

How2Sign is the selected primary dataset for this project. It is the best match for our new goal because it is a continuous American Sign Language dataset with sentence-level clips, English translations, gloss annotations, and multimodal video/keypoint resources.

Use it for:

- understanding the structure of sentence-level ASL data
- testing sentence-window metadata
- comparing our clip schema to a real research dataset
- selecting a few example clips for offline experiments if storage allows
- extracting sampled frames for the CNN baseline
- providing the same test clips for CNN versus VLM comparison

Constraints:

- the full videos are very large
- the dataset is for research purposes
- do not commit downloaded videos to Git
- keep only small metadata samples or derived notes in the repo

Source: https://how2sign.github.io/

## Optional Word-Level Dataset: WLASL

WLASL is useful if the team wants more isolated sign examples for the word-level classifier. It is not enough for sentence translation by itself because it is organized around individual signs.

Use it for:

- improving isolated sign recognition
- expanding beyond the current small vocabulary
- comparing word-level model behavior

Constraints:

- academic/computational use restrictions apply
- it is word-level, not sentence-level
- scraped-video datasets can have availability and consent limitations

Source: https://dxli94.github.io/WLASL/

## Optional Isolated-Sign Dataset: ASL Citizen

ASL Citizen is a strong isolated-sign dataset because it is crowdsourced, consent-based, and recorded in real-world webcam conditions. It is not designed for continuous signing or sentence translation.

Use it for:

- robust isolated-sign recognition
- real-world lighting/background variety
- dictionary-style sign lookup experiments

Constraints:

- it is isolated-sign data, not continuous ASL sentences
- its own recommended-use page cautions against using it as continuous signing data
- the full download is large

Source: https://www.microsoft.com/en-us/research/project/asl-citizen/

## Team-Owned Sentence Dataset

For the class demo, we should record our own small controlled sentence dataset. This gives us examples that exactly match our vocabulary and expected presentation flow.

Suggested starter sentences:

- `HELLO WANT DRINK` -> "Hello, I want a drink."
- `PLEASE HELP` -> "Please help."
- `THANK_YOU FINISHED` -> "Thank you, I am finished."
- `WANT MORE` -> "I want more."
- `NO STOP` -> "No, stop."
- `YES PLEASE` -> "Yes, please."
- `HELLO PLEASE HELP` -> "Hello, please help."
- `EAT MORE` -> "I want more food."

Record each sentence at least 5 times, ideally with more than one teammate signing if possible. Keep raw videos outside Git and commit only metadata.

## Proposed Clip Metadata Format

```json
{
  "clip_id": "team_hello_want_drink_001",
  "split": "test",
  "source": "team-recorded",
  "gloss": ["HELLO", "WANT", "DRINK"],
  "english": "Hello, I want a drink.",
  "video_path": "data/raw/team_hello_want_drink_001.mp4",
  "sampled_frames": [],
  "landmarks_path": null,
  "notes": "Controlled lighting, front-facing signer."
}
```

For CNN training, `sampled_frames` must contain a fixed or pad-able sequence of local frame image paths. The default CNN config expects 16 sampled frames per clip.

## How2Sign Subset Workflow

Use `scripts/create_how2sign_subset.py` to convert downloaded How2Sign metadata into `data/processed/how2sign_subset.jsonl`.

Use `scripts/prepare_clip_dataset.py --extract-frames` to sample frames into `data/interim/frames/`.

Use `scripts/generate_project_results.py` to create report-ready metrics and SVG charts in `results/`.

Large How2Sign videos and extracted frame folders must stay out of Git.

## Storage Rule

Keep large assets out of Git. Store raw video clips in local `data/raw/`, OneDrive, Google Drive, or GitHub Releases if the team needs sharing. Commit only metadata, small samples, and scripts.
