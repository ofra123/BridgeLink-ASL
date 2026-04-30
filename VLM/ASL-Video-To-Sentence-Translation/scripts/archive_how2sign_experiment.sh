#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-archive_500train_eval100}"
BASE_DIR="outputs/how2sign_qwen25vl_3b_qlora"
ARCHIVE_DIR="${BASE_DIR}/${RUN_NAME}"

mkdir -p "${ARCHIVE_DIR}"

echo "Archiving experiment to: ${ARCHIVE_DIR}"

if [ -d "${BASE_DIR}/adapter" ]; then
  rm -rf "${ARCHIVE_DIR}/adapter"
  cp -r "${BASE_DIR}/adapter" "${ARCHIVE_DIR}/adapter"
fi

for file in \
  training_config.yaml \
  train_log.jsonl \
  baseline_metrics.json \
  baseline_predictions.jsonl \
  finetuned_metrics.json \
  finetuned_predictions.jsonl \
  comparison.csv
do
  if [ -f "${BASE_DIR}/${file}" ]; then
    cp "${BASE_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
  fi
done

cat > "${ARCHIVE_DIR}/README.md" <<'EOF'
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
EOF

echo "Archive complete."
find "${ARCHIVE_DIR}" -maxdepth 2 -type f | sort