# Dataset Plan

## Final Dataset Choice

BridgeLink ASL now uses a WLASL-centered dataset strategy:

- Primary training dataset: `WLASL-100`
- Live demo / comparison subset: `WLASL-25`
- Optional team-recorded demo clips for presentation backup

This keeps the project aligned with the final scope: isolated sign recognition
from short video clips, not full continuous ASL translation.

## WLASL-100

WLASL is the main dataset used for model training and benchmarking.

Use it for:

- CNN training on a real word-level ASL benchmark
- Transformer extension experiments
- reporting train / validation / test metrics
- building the main project story around a reproducible, standard dataset

Why it fits:

- word-level labels match the project scope
- the dataset is widely cited in ASL recognition work
- it supports fair comparison with prior methods
- it is practical for landmark-based training on free-tier GPU hardware

## WLASL-25

WLASL-25 is the smaller evaluation subset used for:

- the live demo checkpoint
- the CNN versus VLM reranking comparison
- faster smoke tests on local machines and Hugging Face Space

This subset is easier to present because the label space is smaller and the
predictions are more stable in front of an audience.

## Clip Manifest Format

The current hybrid comparison manifest is:

```text
data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl
```

Each row contains:

- `video_id`
- `true_label`
- `video_path`
- `landmark_path`
- `cnn_top1`
- `cnn_top5`
- `vlm_prompt`

This manifest is the handoff point between the trained CNN and the local VLM
reranking workflow.

## Storage Rule

Keep large raw datasets and extracted landmarks outside Git when possible.
Commit only:

- metadata manifests
- small evaluation subsets
- scripts
- charts
- final metrics

The tracked repo should stay lightweight enough for grading, collaboration,
and Hugging Face deployment.
