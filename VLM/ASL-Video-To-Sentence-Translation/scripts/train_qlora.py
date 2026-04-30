from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.datasets import VideoJsonlDataset
from src.io_utils import append_jsonl, apply_overrides, load_config, output_dir, save_config
from src.qwen_video_utils import load_model_for_training, load_processor, processor_inputs
from src.train_utils import (
    apply_lora,
    build_optimizer,
    build_scheduler,
    oom_help,
    quantization_config_from_config,
)


class QwenVideoSFTCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError(
                "This first-version video collator supports per_device_train_batch_size=1. "
                "Use gradient_accumulation_steps for larger effective batches."
            )
        sample = batch[0]
        messages = sample["messages"]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError(f"No assistant label in training sample id={sample.get('id')}")

        full_inputs = processor_inputs(self.processor, [messages], add_generation_prompt=False)
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        prompt_inputs = processor_inputs(self.processor, [prompt_messages], add_generation_prompt=True)
        labels = full_inputs["input_ids"].clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        labels[full_inputs["input_ids"] == pad_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), model_name=args.model_name, output_dir=args.output_dir)
    out_dir = output_dir(cfg)
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        mixed_precision="bf16" if cfg.get("bf16", True) else ("fp16" if cfg.get("fp16", False) else "no"),
    )
    if not torch.cuda.is_available():
        accelerator.print("Warning: CUDA is unavailable. QLoRA training is expected to require CUDA.")

    processor = load_processor(cfg["model_name"])
    dataset = VideoJsonlDataset(
        cfg["train_jsonl"],
        video_column=cfg["video_column"],
        target_column=cfg["target_column"],
        prompt=cfg["prompt"],
        video_fps=cfg.get("video_fps"),
        max_frames=cfg.get("max_frames"),
        max_samples=args.max_samples or int(cfg.get("train_max_samples", 0)),
        require_target=True,
    )
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty.")

    batch_size = int(cfg.get("per_device_train_batch_size", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=QwenVideoSFTCollator(processor),
    )

    dtype = "bfloat16" if cfg.get("bf16", True) else "float16"
    device_map = {"": accelerator.process_index} if torch.cuda.is_available() else None
    model = load_model_for_training(
        cfg["model_name"],
        quantization_config=quantization_config_from_config(cfg),
        device_map=device_map,
        dtype=dtype,
    )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    if args.adapter_path:
        accelerator.print(f"Continuing training from adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        model = apply_lora(model, cfg)

    if accelerator.is_main_process:
        model.print_trainable_parameters()
        save_config(cfg, out_dir / "training_config.yaml")

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(loader) // int(cfg.get("gradient_accumulation_steps", 8)))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    global_step = 0
    model.train()
    try:
        for epoch in range(int(cfg.get("num_train_epochs", 1))):
            progress = tqdm(loader, disable=not accelerator.is_main_process, desc=f"epoch {epoch + 1}")
            for batch in progress:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    loss_value = float(loss.detach().float().cpu())
                    progress.set_postfix(loss=loss_value, step=global_step)
                    if accelerator.is_main_process:
                        append_jsonl(
                            out_dir / "train_log.jsonl",
                            {"epoch": epoch + 1, "step": global_step, "loss": loss_value},
                        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(oom_help()) from exc

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        print(f"Saved adapter to {adapter_dir}")
        print(f"Saved training config to {out_dir / 'training_config.yaml'}")
        print(f"Saved train log to {out_dir / 'train_log.jsonl'}")


if __name__ == "__main__":
    main()
