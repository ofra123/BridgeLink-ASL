from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.io_utils import apply_overrides, load_config, output_dir, read_jsonl, sample_rows, write_json, write_jsonl
from src.label_utils import normalize_label, normalize_text
from src.metrics import metrics_for_task
from src.qwen_video_utils import generate_one, load_model_for_training, load_processor
from src.train_utils import oom_help, quantization_config_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), model_name=args.model_name, output_dir=args.output_dir)
    out_dir = output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() is False:
        print("Warning: CUDA is unavailable. This may be very slow and 4-bit loading may not work on CPU.", file=sys.stderr)

    rows = sample_rows(read_jsonl(cfg["val_jsonl"]), args.max_samples or int(cfg.get("baseline_max_eval_samples", 0)))
    processor = load_processor(cfg["model_name"])
    model = load_model_for_training(
        cfg["model_name"],
        quantization_config=quantization_config_from_config(cfg),
        device_map="auto",
        dtype="bfloat16" if cfg.get("bf16", True) else "float16",
    )
    model.eval()

    preds = []
    references = []
    predictions = []
    task = cfg["task_name"]
    target_column = cfg["target_column"]
    video_column = cfg["video_column"]
    gen_cfg = cfg.get("generation", {})

    try:
        for row in tqdm(rows, desc="baseline"):
            reference = str(row.get(target_column, ""))
            if not reference:
                raise ValueError(f"Missing target '{target_column}' in row id={row.get('id')}")
            prediction = generate_one(
                model,
                processor,
                row[video_column],
                cfg["prompt"],
                gen_cfg,
                video_fps=cfg.get("video_fps"),
                max_frames=cfg.get("max_frames"),
            )
            norm_ref = normalize_label(reference) if task == "asl_citizen" else normalize_text(reference)
            norm_pred = normalize_label(prediction) if task == "asl_citizen" else normalize_text(prediction)
            correct = norm_ref == norm_pred if task == "asl_citizen" else None
            preds.append(
                {
                    "id": row.get("id"),
                    "video_path": row[video_column],
                    "reference": reference,
                    "prediction": prediction,
                    "normalized_reference": norm_ref,
                    "normalized_prediction": norm_pred,
                    "correct": correct,
                }
            )
            references.append(reference)
            predictions.append(prediction)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(oom_help()) from exc

    metrics = metrics_for_task(task, references, predictions)
    write_jsonl(out_dir / "baseline_predictions.jsonl", preds)
    write_json(out_dir / "baseline_metrics.json", metrics)
    print(metrics)
    print(f"Saved predictions to {out_dir / 'baseline_predictions.jsonl'}")
    print(f"Saved metrics to {out_dir / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
