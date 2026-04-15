# CNN vs VLM Comparison Plan

## Goal

The final comparison is:

```text
trained landmark CNN vs pretrained VLM reranking
```

The CNN is the trained computer vision model. The VLM is evaluated zero-shot as
a visual reasoning model that reranks the CNN's top-5 candidates from the
original signing clip.

## Model A: Landmark CNN

Input:

```text
32 frames x 225 MediaPipe landmark features
```

Architecture:

```text
BatchNorm1d
Conv1d -> BatchNorm1d -> ReLU -> Dropout
Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d
Conv1d -> BatchNorm1d -> ReLU -> Dropout
AdaptiveAvgPool1d
Linear classifier
```

Training defaults:

```text
loss: cross entropy with label smoothing 0.1
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-2
epochs: 50
batch size: 64
```

Outputs:

```text
models/cnn_landmark_best.pt
results/cnn_metrics.json
results/cnn_training_curves.png
results/cnn_confusion_matrix.png
```

## Model B: VLM Reranker

The VLM is not fine-tuned. For each held-out clip, it receives:

```text
video clip
CNN top-5 candidate labels
prompt instructing it to choose only from those labels
```

This tests whether a pretrained vision-language model can improve the CNN's
candidate selection without task-specific training.

## Metrics

Report:

```text
CNN top-1 accuracy
CNN top-5 coverage
VLM-reranked top-5 accuracy
precision / recall / F1 for CNN where available
qualitative VLM successes and failures
```

## Optional Extra: Transformer

The Transformer remains useful for the rubric category covering
Transformers/attention/modern methods. It should be presented as an additional
experiment, not the main CNN vs VLM comparison.
