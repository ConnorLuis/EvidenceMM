from __future__ import annotations

import importlib.metadata
import json

import torch
import transformers


def main() -> int:
    from colpali_engine.models import (
        ColQwen2_5,
        ColQwen2_5_Processor,
    )

    mapping = ColQwen2_5._checkpoint_conversion_mapping
    required_mapping = {
        r"^model\.embed_tokens":
            "language_model.embed_tokens",
        r"^model\.norm":
            "language_model.norm",
    }
    checkpoint_mapping_ok = all(
        mapping.get(key) == value
        for key, value in required_mapping.items()
    )

    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "colpali_engine": importlib.metadata.version(
            "colpali-engine"
        ),
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "gpu_memory_mb": (
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024 / 1024,
                1,
            )
            if torch.cuda.is_available()
            else None
        ),
        "model_class": ColQwen2_5.__name__,
        "processor_class": ColQwen2_5_Processor.__name__,
        "checkpoint_mapping_ok": checkpoint_mapping_ok,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Day 4.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
