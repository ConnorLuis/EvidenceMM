from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image


def load_model_and_processor(
    *,
    model_name: str,
    max_num_visual_tokens: int,
):
    from colpali_engine.models import (
        ColQwen2_5,
        ColQwen2_5_Processor,
    )
    from transformers.utils.import_utils import (
        is_flash_attn_2_available,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    mapping = ColQwen2_5._checkpoint_conversion_mapping
    required = {
        r"^model\.embed_tokens": "language_model.embed_tokens",
        r"^model\.norm": "language_model.norm",
    }
    if not all(
        mapping.get(key) == value
        for key, value in required.items()
    ):
        raise RuntimeError(
            "Unsafe ColQwen2_5 checkpoint mapping"
        )

    started = time.perf_counter()
    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=(
            "flash_attention_2"
            if is_flash_attn_2_available()
            else None
        ),
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(
        model_name,
        max_num_visual_tokens=max_num_visual_tokens,
    )
    load_sec = time.perf_counter() - started

    return model, processor, load_sec


def encode_one_image(
    *,
    model,
    processor,
    image_path: Path,
) -> tuple[torch.Tensor, float]:
    with Image.open(image_path) as image:
        batch = processor.process_images(
            [image.convert("RGB")]
        ).to(model.device)

    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.inference_mode():
        embedding = model(**batch)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return embedding[0].detach().cpu(), elapsed


def encode_query(
    *,
    model,
    processor,
    query: str,
) -> tuple[torch.Tensor, float]:
    batch = processor.process_queries(
        [query]
    ).to(model.device)

    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.inference_mode():
        embedding = model(**batch)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return embedding, elapsed


def score_documents(
    *,
    processor,
    query_embedding: torch.Tensor,
    document_embeddings: list[torch.Tensor],
    device,
) -> torch.Tensor:
    docs = [
        embedding.to(device)
        for embedding in document_embeddings
    ]

    with torch.inference_mode():
        scores = processor.score_multi_vector(
            query_embedding,
            docs,
        )

    return scores[0].detach().float().cpu()


def rank_scores(
    scores: torch.Tensor,
    *,
    top_k: int,
) -> list[tuple[int, float]]:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    top_k = min(top_k, scores.numel())
    order = torch.argsort(
        scores,
        descending=True,
    )[:top_k]

    return [
        (int(index), float(scores[index]))
        for index in order
    ]
