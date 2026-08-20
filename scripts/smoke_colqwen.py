from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
)

from colpali_engine.models import (
    ColQwen2_5,
    ColQwen2_5_Processor,
)
from evidencemm.visual_corpus import load_visual_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = yaml.safe_load(
        (ROOT / "configs/visual_retrieval.yaml").read_text(
            encoding="utf-8"
        )
    )
    pages = load_visual_manifest(
        ROOT / config["page_manifest"]
    )
    if len(pages) != 8:
        raise SystemExit(
            f"expected 8 rendered pages, got {len(pages)}"
        )

    page3 = next(
        page
        for page in pages
        if page.page_number == 3
    )
    image_path = ROOT / page3.image_path

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()

    model = ColQwen2_5.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=(
            "flash_attention_2"
            if is_flash_attn_2_available()
            else None
        ),
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(
        config["model_name"],
        max_num_visual_tokens=int(
            config["max_num_visual_tokens"]
        ),
    )

    load_sec = time.perf_counter() - started

    with Image.open(image_path) as image:
        batch_image = processor.process_images(
            [image.convert("RGB")]
        ).to(model.device)

    batch_query = processor.process_queries(
        ["STS3215 规格书列出的典型工作电压有哪些？"]
    ).to(model.device)

    infer_started = time.perf_counter()

    with torch.inference_mode():
        image_embedding = model(**batch_image)
        query_embedding = model(**batch_query)
        scores = processor.score_multi_vector(
            query_embedding,
            image_embedding,
        )

    inference_sec = time.perf_counter() - infer_started

    payload = {
        "model_name": config["model_name"],
        "page_number": 3,
        "image_embedding_shape": list(
            image_embedding.shape
        ),
        "query_embedding_shape": list(
            query_embedding.shape
        ),
        "score": float(
            scores[0][0].detach().float().cpu()
        ),
        "model_load_sec": load_sec,
        "smoke_inference_sec": inference_sec,
        "peak_allocated_gpu_mb": (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        ),
        "flash_attention_2": (
            is_flash_attn_2_available()
        ),
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
