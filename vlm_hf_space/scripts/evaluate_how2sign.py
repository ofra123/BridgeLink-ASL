from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.io_utils import apply_overrides, load_config, output_dir, read_jsonl, sample_rows, write_json, write_jsonl
from src.label_utils import normalize_text
from src.metrics import how2sign_metrics
from src.qwen_video_utils import generate_one, load_model_for_training, load_processor
from src.train_utils import oom_help, quantization_config_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), model_name=args.model_name, output_dir=args.output_dir)
    if cfg.get("task_name") != "how2sign":
        raise ValueError("evaluate_how2sign.py requires task_name: how2sign")
    out_dir = output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = sample_rows(read_jsonl(cfg["val_jsonl"]), args.max_samples or int(cfg.get("eval_max_samples", 0)))
    processor = load_processor(cfg["model_name"])
    base = load_model_for_training(
        cfg["model_name"],
        quantization_config=quantization_config_from_config(cfg),
        device_map="auto",
        dtype="bfloat16" if cfg.get("bf16", True) else "float16",
    )
    model = PeftModel.from_pretrained(base, args.adapter) if args.adapter else base
    model.eval()

    preds = []
    refs = []
    hyps = []
    suffix = "finetuned" if args.adapter else "baseline"
    try:
        for row in tqdm(rows, desc="eval-how2sign"):
            reference = str(row[cfg["target_column"]])
            prediction = generate_one(
                model,
                processor,
                row[cfg["video_column"]],
                cfg["prompt"],
                cfg.get("generation", {}),
                video_fps=cfg.get("video_fps"),
                max_frames=cfg.get("max_frames"),
            )
            preds.append(
                {
                    "id": row.get("id"),
                    "video_path": row[cfg["video_column"]],
                    "reference": reference,
                    "prediction": prediction,
                    "normalized_reference": normalize_text(reference),
                    "normalized_prediction": normalize_text(prediction),
                    "correct": None,
                }
            )
            refs.append(reference)
            hyps.append(prediction)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(oom_help()) from exc

    metrics = how2sign_metrics(refs, hyps)
    write_jsonl(out_dir / f"{suffix}_predictions.jsonl", preds)
    write_json(out_dir / f"{suffix}_metrics.json", metrics)
    print(metrics)
    print(f"Saved predictions to {out_dir / f'{suffix}_predictions.jsonl'}")
    print(f"Saved metrics to {out_dir / f'{suffix}_metrics.json'}")


if __name__ == "__main__":
    main()
