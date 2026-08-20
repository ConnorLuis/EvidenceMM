from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.temporal_evidence import EpisodeManifest, load_frame_records
from evidencemm.temporal_slicing import (
    build_temporal_slices,
    save_temporal_slices,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(
        (ROOT / "configs/robot_sequence_evidence.yaml").read_text(
            encoding="utf-8"
        )
    )

    manifest = EpisodeManifest.model_validate_json(
        (
            ROOT
            / config["manifest_root"]
            / f"{args.episode_id}.json"
        ).read_text(encoding="utf-8")
    )

    processed_dir = ROOT / config["processed_root"] / args.episode_id
    records = load_frame_records(processed_dir / "frames.jsonl")

    slices = build_temporal_slices(
        manifest=manifest,
        records=records,
        slice_duration_sec=float(config["slice_duration_sec"]),
    )

    output = processed_dir / "temporal_slices.jsonl"
    save_temporal_slices(slices, output)

    midpoint_errors = [item.midpoint_error_ms for item in slices]

    print(
        json.dumps(
            {
                "episode_id": manifest.episode_id,
                "timestamp_source": manifest.timestamp_source,
                "slice_duration_sec": float(
                    config["slice_duration_sec"]
                ),
                "slice_count": len(slices),
                "first_slice": slices[0].model_dump(),
                "last_slice": slices[-1].model_dump(),
                "mean_midpoint_error_ms": (
                    sum(midpoint_errors) / len(midpoint_errors)
                ),
                "max_midpoint_error_ms": max(midpoint_errors),
                "reencoded_images": 0,
                "output": output.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
