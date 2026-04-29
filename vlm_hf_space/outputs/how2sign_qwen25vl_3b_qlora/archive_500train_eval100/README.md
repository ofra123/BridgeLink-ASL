# How2Sign Qwen2.5-VL-3B QLoRA Experiment

Model: Qwen/Qwen2.5-VL-3B-Instruct

Task: How2Sign continuous ASL video to English translation

Fine-tuning method: QLoRA / LoRA adapter

Base model weights modified: No. Only adapter weights were trained.

This archive contains one experiment snapshot. Later training/evaluation runs may overwrite the live files in the parent output directory, so this folder preserves the adapter, metrics, predictions, config, and training log from this run.

Files:
- adapter/: trained LoRA adapter
- training_config.yaml: training configuration
- train_log.jsonl: training loss log
- baseline_metrics.json: base model metrics
- baseline_predictions.jsonl: base model predictions
- finetuned_metrics.json: fine-tuned adapter metrics
- finetuned_predictions.jsonl: fine-tuned adapter predictions
- comparison.csv: before/after metric comparison
