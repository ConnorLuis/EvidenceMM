from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

import torch
import transformers
import yaml

from evidencemm.data_binding import sha256_file
from evidencemm.visual_corpus import load_visual_manifest
from evidencemm.visual_retrieval import (
    encode_one_image,
    load_model_and_processor,
)


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
            f"expected 8 pages, got {len(pages)}"
        )

    image_paths = []
    for page in pages:
        path = ROOT / page.image_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != page.image_sha256:
            raise SystemExit(
                f"page image SHA256 drift: {page.image_path}"
            )
        image_paths.append(path)

    model, processor, model_load_sec = (
        load_model_and_processor(
            model_name=config["model_name"],
            max_num_visual_tokens=int(
                config["max_num_visual_tokens"]
            ),
        )
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    embeddings = []
    page_encode_secs = []

    started = time.perf_counter()
    for page, image_path in zip(pages, image_paths):
        embedding, elapsed = encode_one_image(
            model=model,
            processor=processor,
            image_path=image_path,
        )
        embeddings.append(embedding)
        page_encode_secs.append(elapsed)
        print(
            f"encoded page={page.page_number} "
            f"shape={list(embedding.shape)} "
            f"sec={elapsed:.4f}"
        )

    index_encode_sec = time.perf_counter() - started
    peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    output = ROOT / config["index_file"]
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "format_version": 1,
        "mode": "vision_only_colqwen2_5",
        "model_name": config["model_name"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "colpali_engine_version": importlib.metadata.version(
            "colpali-engine"
        ),
        "max_num_visual_tokens": int(
            config["max_num_visual_tokens"]
        ),
        "render_dpi": int(config["render_dpi"]),
        "pages": [
            page.to_dict()
            for page in pages
        ],
        "embeddings": embeddings,
        "stats": {
            "model_load_sec": model_load_sec,
            "index_encode_sec": index_encode_sec,
            "page_encode_secs": page_encode_secs,
            "peak_allocated_gpu_mb": peak_mb,
        },
    }
    torch.save(payload, output)

    print(
        json.dumps(
            {
                "model_name": payload["model_name"],
                "page_count": len(pages),
                "embedding_shapes": [
                    list(item.shape)
                    for item in embeddings
                ],
                **payload["stats"],
                "index_file": str(
                    output.relative_to(ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
