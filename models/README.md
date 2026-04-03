# Model Artifacts

This folder stores saved baseline models and later exported production-ready artifacts.

Current starter format:

- `model_type`: `centroid-baseline`
- `feature_length`: number of landmark features per sample
- `labels`: label-to-centroid mapping with sample counts

Suggested lifecycle:

1. save early centroid baselines here during phase 2
2. compare them against later TensorFlow or TensorFlow Lite exports
3. keep only small reproducible artifacts in Git
