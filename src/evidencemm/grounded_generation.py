from __future__ import annotations

import time

import torch


def load_generator(
    *,
    model_name: str,
    min_visual_tokens_per_image: int,
    max_visual_tokens_per_image: int,
    attn_implementation: str,
):
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Day 6 generator baseline requires CUDA"
        )

    if min_visual_tokens_per_image < 1:
        raise ValueError(
            "min_visual_tokens_per_image must be >= 1"
        )
    if (
        max_visual_tokens_per_image
        < min_visual_tokens_per_image
    ):
        raise ValueError(
            "max visual tokens must be >= min visual tokens"
        )

    started = time.perf_counter()

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation=attn_implementation,
        )
        .eval()
    )

    processor = AutoProcessor.from_pretrained(
        model_name
    )

    processor.image_processor.size = {
        "shortest_edge":
            min_visual_tokens_per_image
            * 32
            * 32,
        "longest_edge":
            max_visual_tokens_per_image
            * 32
            * 32,
    }

    load_sec = time.perf_counter() - started
    return model, processor, load_sec


def generate_grounded(
    *,
    model,
    processor,
    messages,
    max_new_tokens: int,
) -> tuple[str, float]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    trimmed = generated[:, input_length:]
    output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output, elapsed
