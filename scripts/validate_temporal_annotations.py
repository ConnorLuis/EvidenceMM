from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.temporal_evidence import (
    EpisodeManifest,
    load_frame_records,
)
from evidencemm.temporal_eval import (
    load_annotations,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-id",
        required=True,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
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

    annotation_path = (
        ROOT
        / "data/eval"
        / (
            "day7_temporal_events_"
            f"{args.episode_id}.jsonl"
        )
    )
    annotations = load_annotations(
        annotation_path
    )

    if any(
        item.episode_id != manifest.episode_id
        for item in annotations
    ):
        raise SystemExit(
            "annotation episode_id mismatch"
        )

    verified = [
        item
        for item in annotations
        if item.status == "verified"
    ]
    drafts = [
        item
        for item in annotations
        if item.status == "draft"
    ]

    valid_indices = {
        item.frame_index
        for item in records
        if item.camera == "front"
    }

    for item in verified:
        assert item.start_frame_index is not None
        assert item.end_frame_index_inclusive is not None

        if item.start_frame_index not in valid_indices:
            raise SystemExit(
                f"{item.event_id} invalid start frame"
            )
        if (
            item.end_frame_index_inclusive
            not in valid_indices
        ):
            raise SystemExit(
                f"{item.event_id} invalid end frame"
            )

    print(
        json.dumps(
            {
                "episode_id":
                    manifest.episode_id,
                "annotations":
                    len(annotations),
                "verified":
                    len(verified),
                "draft":
                    len(drafts),
                "all_verified_bounds_valid":
                    True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
