from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evidencemm.schemas import BaselineRecord


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "baseline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct Qwen3-VL baseline without retrieval"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def local_uri(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path.as_uri()


def main() -> int:
    args = parse_args()

    if not args.image and not args.video:
        raise SystemExit("provide at least one --image or --video")

    config = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )

    model_name = config["model_name"]
    image_patch_size = int(config.get("image_patch_size", 16))
    max_new_tokens = int(config.get("max_new_tokens", 256))
    min_pixels = int(config.get("min_pixels", 50176))
    max_pixels = int(config.get("max_pixels", 1003520))
    video_fps = float(config.get("video_fps", 1.0))

    content: list[dict] = []
    input_files: list[str] = []

    for image in args.image:
        resolved = Path(image).expanduser().resolve()
        uri = local_uri(image)
        input_files.append(str(resolved))
        content.append(
            {
                "type": "image",
                "image": uri,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )

    for video in args.video:
        resolved = Path(video).expanduser().resolve()
        uri = local_uri(video)
        input_files.append(str(resolved))
        content.append(
            {
                "type": "video",
                "video": uri,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": video_fps,
            }
        )

    content.append(
        {
            "type": "text",
            "text": args.question,
        }
    )
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    print(f"loading model: {model_name}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=image_patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)
    else:
        video_metadata = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )

    # Official examples place processed tensors on the model device.
    inputs = inputs.to(model.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    peak_mb = None
    if torch.cuda.is_available():
        peak_mb = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )

    record = BaselineRecord(
        model_name=model_name,
        question=args.question,
        input_files=input_files,
        response=response,
        latency_sec=latency,
        peak_gpu_memory_mb=peak_mb,
    )

    if args.output:
        output = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        output = (
            ROOT
            / "reports"
            / "baseline"
            / f"{stamp}.json"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== response ===")
    print(response)

    print("\n=== metrics ===")
    print(f"latency_sec={latency:.3f}")
    print(f"peak_gpu_memory_mb={peak_mb}")
    print(f"report={output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
