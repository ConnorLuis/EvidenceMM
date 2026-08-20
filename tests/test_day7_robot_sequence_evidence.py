from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidencemm.temporal_evidence import (
    CameraSpec,
    EpisodeManifest,
    FrameRecord,
    canonical_episode_hash,
)


def frame(
    *,
    camera: str,
    index: int,
    sha: str,
) -> FrameRecord:
    return FrameRecord(
        episode_id="ep0",
        camera=camera,
        frame_index=index,
        timestamp_sec=index / 15.0,
        sample_timestamp_ns=1000 + index,
        source_timestamp_ns=900 + index,
        source_age_ms=5.0,
        camera_sequence=index,
        image_relpath=f"{camera}/{index:06d}.jpg",
        image_sha256=sha,
        width_px=640,
        height_px=480,
    )


def test_episode_manifest_camera_contract():
    manifest = EpisodeManifest(
        episode_id="ep0",
        source_schema_version="source-v1",
        source_script_version="rec-v5",
        task="pick",
        frame_count=2,
        nominal_hz=15.0,
        actual_record_span_seconds=1.0,
        timestamp_source="samples_csv.elapsed_ns",
        metadata_sha256="a" * 64,
        samples_csv_sha256="b" * 64,
        episode_sha256="c" * 64,
        cameras=[
            CameraSpec(
                camera="front",
                frame_count=2,
                width_px=640,
                height_px=480,
                transform="none",
            ),
            CameraSpec(
                camera="wrist",
                frame_count=2,
                width_px=480,
                height_px=640,
                transform="ccw90",
            ),
        ],
        source_checks_overall_pass=True,
    )
    assert manifest.cameras[1].transform == "ccw90"


def test_episode_manifest_rejects_wrong_camera_order():
    with pytest.raises(ValidationError):
        EpisodeManifest(
            episode_id="ep0",
            source_schema_version="source-v1",
            source_script_version="rec-v5",
            task="pick",
            frame_count=1,
            nominal_hz=15.0,
            actual_record_span_seconds=1.0,
            timestamp_source="samples_csv.elapsed_ns",
            metadata_sha256="a" * 64,
            samples_csv_sha256="b" * 64,
            episode_sha256="c" * 64,
            cameras=[
                CameraSpec(
                    camera="wrist",
                    frame_count=1,
                    width_px=480,
                    height_px=640,
                    transform="ccw90",
                ),
                CameraSpec(
                    camera="front",
                    frame_count=1,
                    width_px=640,
                    height_px=480,
                    transform="none",
                ),
            ],
            source_checks_overall_pass=True,
        )


def test_episode_hash_is_order_independent():
    records = [
        frame(
            camera="front",
            index=0,
            sha="1" * 64,
        ),
        frame(
            camera="wrist",
            index=0,
            sha="2" * 64,
        ),
    ]
    first = canonical_episode_hash(
        metadata_sha256="a" * 64,
        samples_csv_sha256="b" * 64,
        records=records,
    )
    second = canonical_episode_hash(
        metadata_sha256="a" * 64,
        samples_csv_sha256="b" * 64,
        records=list(reversed(records)),
    )
    assert first == second


def test_episode_hash_changes_when_frame_changes():
    first = canonical_episode_hash(
        metadata_sha256="a" * 64,
        samples_csv_sha256="b" * 64,
        records=[
            frame(
                camera="front",
                index=0,
                sha="1" * 64,
            )
        ],
    )
    second = canonical_episode_hash(
        metadata_sha256="a" * 64,
        samples_csv_sha256="b" * 64,
        records=[
            frame(
                camera="front",
                index=0,
                sha="2" * 64,
            )
        ],
    )
    assert first != second
