from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.temporal_evidence import EpisodeManifest, load_frame_records
from evidencemm.temporal_slicing import load_temporal_slices


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
    slices = load_temporal_slices(
        processed_dir / "temporal_slices.jsonl"
    )

    record_map = {
        (record.camera, record.frame_index): record
        for record in records
    }

    if not slices:
        raise SystemExit("no temporal slices found")

    previous_end = None

    for index, item in enumerate(slices):
        expected_id = (
            f"{manifest.episode_id}_slice_{index:04d}"
        )
        if item.slice_group_id != expected_id:
            raise SystemExit("slice_group_id sequence mismatch")

        if previous_end is not None:
            if abs(item.start_sec - previous_end) > 1e-12:
                raise SystemExit(
                    "temporal windows are not contiguous"
                )
        previous_end = item.end_sec

        for camera_item in item.cameras:
            key = (
                camera_item.camera,
                item.midpoint_frame_index,
            )
            record = record_map.get(key)
            if record is None:
                raise SystemExit("midpoint FrameRecord missing")

            checks = [
                camera_item.image_relpath == record.image_relpath,
                camera_item.image_sha256 == record.image_sha256,
                (
                    camera_item.source_timestamp_ns
                    == record.source_timestamp_ns
                ),
                (
                    abs(
                        camera_item.source_age_ms
                        - record.source_age_ms
                    )
                    < 1e-12
                ),
            ]
            if not all(checks):
                raise SystemExit(
                    "midpoint evidence does not match original FrameRecord"
                )

    print(
        json.dumps(
            {
                "episode_id": manifest.episode_id,
                "slice_count": len(slices),
                "pairwise_midpoints_valid": True,
                "original_frame_hashes_reused": True,
                "reencoded_images": 0,
                "first_midpoint_frame": (
                    slices[0].midpoint_frame_index
                ),
                "last_midpoint_frame": (
                    slices[-1].midpoint_frame_index
                ),
                "timestamp_source": manifest.timestamp_source,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
