from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.data_binding import sha256_file
from evidencemm.temporal_evidence import (
    CameraSpec,
    EpisodeManifest,
    build_frame_records,
    canonical_episode_hash,
    load_sample_rows,
    load_source_metadata,
    save_frame_records,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--episode-id", default=None)
    args = parser.parse_args()

    episode_dir = Path(
        args.episode_dir
    ).expanduser().resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(episode_dir)

    episode_id = args.episode_id or episode_dir.name

    config = yaml.safe_load(
        (
            ROOT / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.csv"
    metadata = load_source_metadata(metadata_path)
    rows = load_sample_rows(samples_path)

    if metadata["status"] != "completed":
        raise SystemExit("source episode is not completed")
    if not metadata["valid_for_training"]:
        raise SystemExit("source episode is not valid_for_training")
    if not metadata["overall_pass"]:
        raise SystemExit("source episode overall_pass is false")

    expected_count = int(
        metadata["settings"]["expected_sample_count"]
    )
    if len(rows) != expected_count:
        raise SystemExit(
            f"sample count mismatch: {len(rows)} != {expected_count}"
        )

    for camera in ("front", "wrist"):
        count = int(
            metadata["results"][f"{camera}_image_count"]
        )
        if count != expected_count:
            raise SystemExit(
                f"metadata {camera}_image_count mismatch"
            )

    records = build_frame_records(
        episode_dir=episode_dir,
        episode_id=episode_id,
        rows=rows,
    )

    front = [
        item for item in records if item.camera == "front"
    ]
    wrist = [
        item for item in records if item.camera == "wrist"
    ]

    front_dims = {
        (item.width_px, item.height_px)
        for item in front
    }
    wrist_dims = {
        (item.width_px, item.height_px)
        for item in wrist
    }
    if len(front_dims) != 1:
        raise SystemExit(f"front dimension drift: {front_dims}")
    if len(wrist_dims) != 1:
        raise SystemExit(f"wrist dimension drift: {wrist_dims}")

    metadata_sha = sha256_file(metadata_path)
    samples_sha = sha256_file(samples_path)
    episode_sha = canonical_episode_hash(
        metadata_sha256=metadata_sha,
        samples_csv_sha256=samples_sha,
        records=records,
    )

    front_width, front_height = next(iter(front_dims))
    wrist_width, wrist_height = next(iter(wrist_dims))

    manifest = EpisodeManifest(
        episode_id=episode_id,
        source_schema_version=metadata["schema_version"],
        source_script_version=metadata["script_version"],
        task=metadata["task"],
        frame_count=expected_count,
        nominal_hz=float(metadata["settings"]["hz"]),
        actual_record_span_seconds=float(
            metadata["timing"]["actual_record_span_seconds"]
        ),
        timestamp_source=config["timestamp_source"],
        metadata_sha256=metadata_sha,
        samples_csv_sha256=samples_sha,
        episode_sha256=episode_sha,
        cameras=[
            CameraSpec(
                camera="front",
                frame_count=expected_count,
                width_px=front_width,
                height_px=front_height,
                transform=metadata["settings"][
                    "camera_transforms"
                ]["front"],
            ),
            CameraSpec(
                camera="wrist",
                frame_count=expected_count,
                width_px=wrist_width,
                height_px=wrist_height,
                transform=metadata["settings"][
                    "camera_transforms"
                ]["wrist"],
            ),
        ],
        source_checks_overall_pass=bool(
            metadata["overall_pass"]
        ),
    )

    processed_dir = (
        ROOT / config["processed_root"] / episode_id
    )
    frames_path = processed_dir / "frames.jsonl"
    save_frame_records(records, frames_path)

    manifest_path = (
        ROOT
        / config["manifest_root"]
        / f"{episode_id}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    source_skews_ms = [
        abs(
            row.front_source_timestamp_ns
            - row.wrist_source_timestamp_ns
        ) / 1e6
        for row in rows
    ]

    print(
        json.dumps(
            {
                "episode_id": episode_id,
                "frame_count": expected_count,
                "frame_records": len(records),
                "timestamp_source":
                    config["timestamp_source"],
                "nominal_hz": manifest.nominal_hz,
                "actual_record_span_seconds":
                    manifest.actual_record_span_seconds,
                "front_dimensions":
                    [front_width, front_height],
                "wrist_dimensions":
                    [wrist_width, wrist_height],
                "front_transform":
                    manifest.cameras[0].transform,
                "wrist_transform":
                    manifest.cameras[1].transform,
                "max_camera_source_skew_ms":
                    max(source_skews_ms),
                "mean_camera_source_skew_ms":
                    sum(source_skews_ms)
                    / len(source_skews_ms),
                "metadata_sha256": metadata_sha,
                "samples_csv_sha256": samples_sha,
                "episode_sha256": episode_sha,
                "manifest_path":
                    manifest_path.relative_to(ROOT).as_posix(),
                "frames_index_path":
                    frames_path.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
