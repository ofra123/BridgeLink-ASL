from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from .io_utils import resolve_path
from .label_utils import normalize_text
from .train_utils import str_to_torch_dtype


def load_processor(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor


def _load_model(model_name: str, **kwargs: Any):
    auto_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_cls is None:
        auto_cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if auto_cls is None:
        auto_cls = getattr(transformers, "AutoModelForCausalLM", None)
    if auto_cls is None:
        raise RuntimeError("No compatible Transformers auto model class is available.")
    try:
        return auto_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the multimodal model with Transformers auto classes. "
            "Check that your transformers version supports the configured model_name "
            f"({model_name}). Original error: {exc}"
        ) from exc


def load_model_for_inference(
    model_name: str,
    dtype: str | torch.dtype | None = None,
    device_map: str | dict[str, Any] | None = "auto",
):
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else str_to_torch_dtype(dtype or "bfloat16")
    model = _load_model(model_name, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()
    return model


def load_model_for_training(
    model_name: str,
    quantization_config: Any = None,
    device_map: str | dict[str, Any] | None = "auto",
    dtype: str | torch.dtype | None = None,
):
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else str_to_torch_dtype(dtype or "bfloat16")
    return _load_model(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
    )


# def video_content(video_path: str | Path, video_fps: int | float | None = None, max_frames: int | None = None) -> dict[str, Any]:
#     path = resolve_path(video_path)
#     if not path.exists():
#         raise FileNotFoundError(f"Video file not found: {path}")

#     item: dict[str, Any] = {"type": "video", "video": str(path)}

#     # qwen-vl-utils accepts either fps OR nframes, not both.
#     # Prefer nframes because it gives predictable GPU memory usage.
#     if max_frames is not None and max_frames > 0:
#         item["nframes"] = int(max_frames)
#     elif video_fps is not None:
#         item["fps"] = video_fps

#     return item
def video_content(video_path: str | Path, video_fps: int | float | None = None, max_frames: int | None = None) -> dict[str, Any]:
    path = resolve_path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    item: dict[str, Any] = {"type": "video", "video": str(path)}

    # qwen-vl-utils accepts either fps OR nframes, not both.
    # Prefer nframes for predictable memory, but cap it to the actual video length
    # so short clips do not crash.
    if max_frames is not None and max_frames > 0:
        nframes = int(max_frames)

        try:
            from decord import VideoReader, cpu

            vr = VideoReader(str(path), ctx=cpu(0))
            total_frames = len(vr)

            if total_frames > 0:
                nframes = min(nframes, total_frames)

                # qwen-vl-utils usually wants at least 2 frames.
                if nframes < 2:
                    nframes = 2

                # Qwen video frame counts are safest as even numbers.
                if nframes > 2 and nframes % 2 != 0:
                    nframes -= 1
        except Exception:
            # If frame counting fails, fall back to requested max_frames.
            pass

        item["nframes"] = nframes

    elif video_fps is not None:
        item["fps"] = video_fps

    return item


def build_messages(
    video_path: str | Path,
    prompt: str,
    target: str | None = None,
    *,
    video_fps: int | float | None = None,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                video_content(video_path, video_fps=video_fps, max_frames=max_frames),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if target is not None:
        target = str(target).strip()
        if not target:
            raise ValueError("No assistant label/translation was provided for the training sample.")
        messages.append({"role": "assistant", "content": target})
    return messages


def _drop_nframes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        content = []
        for part in msg.get("content", []):
            if isinstance(part, dict):
                part = dict(part)
                part.pop("nframes", None)
            content.append(part)
        cleaned.append({**msg, "content": content})
    return cleaned


def process_messages(messages: list[dict[str, Any]]):
    try:
        return process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs, {}
    except Exception as exc:
        if "nframes" in str(exc).lower():
            return process_messages(_drop_nframes(messages))
        raise RuntimeError(
            "Video preprocessing failed. Check that decord is installed, the video format is supported, "
            "and the file path is valid. Original error: "
            f"{exc}"
        ) from exc


def processor_inputs(
    processor,
    messages_batch: list[list[dict[str, Any]]],
    *,
    add_generation_prompt: bool,
    padding: bool = True,
) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    image_inputs_batch = []
    video_inputs_batch = []
    merged_video_kwargs: dict[str, Any] = {}

    for messages in messages_batch:
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )

        image_inputs, video_inputs, video_kwargs = process_messages(messages)

        if image_inputs is not None:
            image_inputs_batch.append(image_inputs)

        if video_inputs is not None:
            video_inputs_batch.append(video_inputs)

        merged_video_kwargs.update(video_kwargs or {})

    processor_kwargs = {
        "text": texts,
        "padding": padding,
        "return_tensors": "pt",
        **merged_video_kwargs,
    }

    if image_inputs_batch:
        processor_kwargs["images"] = image_inputs_batch

    if video_inputs_batch:
        processor_kwargs["videos"] = video_inputs_batch

    return processor(**processor_kwargs)


def move_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


@torch.no_grad()
def generate_one(
    model,
    processor,
    video_path: str | Path,
    prompt: str,
    generation_config: dict[str, Any] | None = None,
    *,
    video_fps: int | float | None = None,
    max_frames: int | None = None,
) -> str:
    messages = build_messages(video_path, prompt, video_fps=video_fps, max_frames=max_frames)
    inputs = processor_inputs(processor, [messages], add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = move_to_device(inputs, device)
    gen_kwargs = dict(generation_config or {})
    if gen_kwargs.get("do_sample") is False and "temperature" in gen_kwargs:
        gen_kwargs.pop("temperature", None)
    generated_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, input_len:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return normalize_prediction(text)


def normalize_prediction(text: str) -> str:
    return normalize_text(text).strip()
