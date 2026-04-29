from __future__ import annotations

import math
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, get_cosine_schedule_with_warmup


def str_to_torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    value = str(name).lower()
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def quantization_config_from_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not config.get("load_in_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=str_to_torch_dtype(config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
    )


def apply_lora(model: torch.nn.Module, config: dict[str, Any]) -> torch.nn.Module:
    if config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )
    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=list(config.get("target_modules", [])),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found. Check LoRA target_modules.")
    return torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], steps_per_epoch: int):
    epochs = int(config.get("num_train_epochs", 1))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = math.ceil(total_steps * float(config.get("warmup_ratio", 0.03)))
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def oom_help() -> str:
    return (
        "CUDA out of memory. Try reducing max_frames, train_max_samples, or "
        "per_device_train_batch_size; use a smaller model_name; keep load_in_4bit true; "
        "or run on a GPU with more VRAM."
    )
