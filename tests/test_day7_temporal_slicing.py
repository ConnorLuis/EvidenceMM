from __future__ import annotations

from evidencemm.temporal_evidence import (
    CameraSpec,
    EpisodeManifest,
    FrameRecord,
)
from evidencemm.temporal_slicing import build_temporal_slices


def manifest(frame_count: int) -> EpisodeManifest:
    return EpisodeManifest(
        episode_id="ep0",
        source_schema_version="source-v1",
        source_script_version="rec-v5",
        task="pick",
        frame_count=frame_count,
        nominal_hz=2.0,
        actual_record_span_seconds=4.0,
        timestamp_source="samples_csv.elapsed_ns",
        metadata_sha256="a" * 64,
        samples_csv_sha256="b" * 64,
        episode_sha256="c" * 64,
        cameras=[
            CameraSpec(
                camera="front",
                frame_count=frame_count,
                width_px=640,
                height_px=480,
                transform="none",
            ),
            CameraSpec(
                camera="wrist",
                frame_count=frame_count,
                width_px=480,
                height_px=640,
                transform="ccw90",
            ),
        ],
        source_checks_overall_pass=True,
    )


def records(timestamps: list[float]) -> list[FrameRecord]:
    output = []

    for index, timestamp in enumerate(timestamps):
        for camera in ("front", "wrist"):
            output.append(
                FrameRecord(
                    episode_id="ep0",
                    camera=camera,
                    frame_index=index,
                    timestamp_sec=timestamp,
                    sample_timestamp_ns=1_000_000_000 + index,
                    source_timestamp_ns=900_000_000 + index,
                    source_age_ms=5.0,
                    camera_sequence=index,
                    image_relpath=(
                        f"{camera}/{index:06d}.jpg"
                    ),
                    image_sha256=(
                        ("1" if camera == "front" else "2") * 64
                    ),
                    width_px=(
                        640 if camera == "front" else 480
                    ),
                    height_px=(
                        480 if camera == "front" else 640
                    ),
                )
            )

    return output


def test_build_two_second_slices_from_real_timestamps():
    timestamps = [
        0.01,
        0.51,
        1.01,
        1.51,
        2.01,
        2.51,
        3.01,
        3.51,
    ]

    result = build_temporal_slices(
        manifest=manifest(len(timestamps)),
        records=records(timestamps),
        slice_duration_sec=2.0,
    )

    assert len(result) == 2
    assert result[0].start_sec == 0.0
    assert result[0].end_sec == 2.0
    assert result[1].start_sec == 2.0
    assert result[1].end_sec == 4.0


def test_midpoint_uses_nearest_real_sample():
    timestamps = [0.1, 0.7, 1.2, 1.8]

    result = build_temporal_slices(
        manifest=manifest(len(timestamps)),
        records=records(timestamps),
        slice_duration_sec=2.0,
    )

    assert result[0].midpoint_target_sec == 1.0
    assert result[0].midpoint_frame_index == 2
    assert result[0].midpoint_timestamp_sec == 1.2
    assert abs(result[0].midpoint_error_ms - 200.0) < 1e-9


def test_midpoint_tie_breaks_to_lower_frame_index():
    timestamps = [0.5, 1.5]

    result = build_temporal_slices(
        manifest=manifest(len(timestamps)),
        records=records(timestamps),
        slice_duration_sec=2.0,
    )

    assert result[0].midpoint_frame_index == 0


def test_both_cameras_reuse_same_frame_index():
    timestamps = [0.1, 1.0, 1.9]

    result = build_temporal_slices(
        manifest=manifest(len(timestamps)),
        records=records(timestamps),
        slice_duration_sec=2.0,
    )

    item = result[0]
    assert [
        camera.camera for camera in item.cameras
    ] == ["front", "wrist"]
    assert all(
        camera.frame_index == item.midpoint_frame_index
        for camera in item.cameras
    )


def test_original_frame_hashes_are_reused():
    timestamps = [0.1, 1.0, 1.9]

    source = records(timestamps)
    result = build_temporal_slices(
        manifest=manifest(len(timestamps)),
        records=source,
        slice_duration_sec=2.0,
    )

    source_hashes = {
        (record.camera, record.frame_index): record.image_sha256
        for record in source
    }

    item = result[0]
    for camera in item.cameras:
        assert (
            camera.image_sha256
            == source_hashes[
                (
                    camera.camera,
                    item.midpoint_frame_index,
                )
            ]
        )
