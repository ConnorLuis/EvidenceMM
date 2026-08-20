from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.temporal_evidence import (
    EpisodeManifest,
    load_frame_records,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    manifest = EpisodeManifest.model_validate_json(
        (
            ROOT
            / config["manifest_root"]
            / f"{args.episode_id}.json"
        ).read_text(encoding="utf-8")
    )

    records = load_frame_records(
        ROOT
        / config["processed_root"]
        / args.episode_id
        / "frames.jsonl"
    )

    expected = manifest.frame_count * len(manifest.cameras)
    if len(records) != expected:
        raise SystemExit(
            f"FrameRecord count mismatch: "
            f"{len(records)} != {expected}"
        )

    for camera in ("front", "wrist"):
        camera_records = [
            record
            for record in records
            if record.camera == camera
        ]
        indices = [
            record.frame_index
            for record in camera_records
        ]
        if indices != list(range(manifest.frame_count)):
            raise SystemExit(
                f"{camera} frame indices not contiguous"
            )

    pair_map = {}
    for record in records:
        pair_map.setdefault(
            record.frame_index,
            set(),
        ).add(record.camera)

    incomplete = [
        index
        for index, cameras in pair_map.items()
        if cameras != {"front", "wrist"}
    ]
    if incomplete:
        raise SystemExit(
            "incomplete sample-synchronized pairs: "
            + repr(incomplete[:20])
        )

    print(
        json.dumps(
            {
                "episode_id": manifest.episode_id,
                "frame_count": manifest.frame_count,
                "frame_records": len(records),
                "pair_count": len(pair_map),
                "pairing_complete": True,
                "timestamp_source":
                    manifest.timestamp_source,
                "episode_sha256":
                    manifest.episode_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
