from __future__ import annotations

import json
import platform

import torch
import transformers


def main() -> int:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_vram_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 2)
            if torch.cuda.is_available()
            else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
