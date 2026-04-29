from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.io_utils import apply_overrides, load_config
from src.qwen_video_utils import generate_one, load_model_for_training, load_processor
from src.train_utils import oom_help, quantization_config_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--video_fps", type=float, default=None)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), model_name=args.model_name)
    processor = load_processor(cfg["model_name"])
    model = load_model_for_training(
        cfg["model_name"],
        quantization_config=quantization_config_from_config(cfg),
        device_map="auto",
        dtype="bfloat16" if cfg.get("bf16", True) else "float16",
    )
    model.eval()
    try:
        prediction = generate_one(
            model,
            processor,
            args.video,
            args.prompt or cfg["prompt"],
            cfg.get("generation", {}),
            video_fps=args.video_fps if args.video_fps is not None else cfg.get("video_fps"),
            max_frames=args.max_frames if args.max_frames is not None else cfg.get("max_frames"),
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(oom_help()) from exc
    print(prediction)


if __name__ == "__main__":
    main()
