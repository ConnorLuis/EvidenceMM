from __future__ import annotations

import pytest

from evidencemm.multi_keyframe_selection import (
    build_multi_keyframe_selection,
    interior_quantile_fractions,
)
from evidencemm.temporal_evidence import FrameRecord
from evidencemm.temporal_slicing import TemporalSlice


def pair(
    frame_index: int,
    timestamp_sec: float,
) -> list[FrameRecord]:
    result = []
    for camera in ("front", "wrist"):
        result.append(
            FrameRecord(
                episode_id="ep0",
                camera=camera,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                sample_timestamp_ns=1_000_000 + frame_index,
                source_timestamp_ns=2_000_000 + frame_index,
                source_age_ms=1.0,
                camera_sequence=frame_index,
                image_relpath=(
                    f"{camera}/{frame_index:06d}.jpg"
                ),
                image_sha256=(
                    ("1" if camera == "front" else "2")
                    * 64
                ),
                width_px=640,
                height_px=480,
            )
        )
    return result


def records(
    timestamps: list[float],
) -> list[FrameRecord]:
    result = []
    for index, timestamp in enumerate(timestamps):
        result.extend(pair(index, timestamp))
    return result


def temporal_slice() -> TemporalSlice:
    recs = records([0.0, 0.5, 1.0, 1.5])
    front = [item for item in recs if item.camera == "front"][2]
    wrist = [item for item in recs if item.camera == "wrist"][2]

    cameras = []
    from evidencemm.temporal_slicing import CameraMidpointEvidence

    for item in (front, wrist):
        cameras.append(
            CameraMidpointEvidence(
                camera=item.camera,
                frame_index=item.frame_index,
                timestamp_sec=item.timestamp_sec,
                source_timestamp_ns=item.source_timestamp_ns,
                source_age_ms=item.source_age_ms,
                camera_sequence=item.camera_sequence,
                image_relpath=item.image_relpath,
                image_sha256=item.image_sha256,
                width_px=item.width_px,
                height_px=item.height_px,
            )
        )

    return TemporalSlice(
        slice_group_id="ep0_slice_0000",
        episode_id="ep0",
        start_sec=0.0,
        end_sec=2.0,
        start_frame_index=0,
        end_frame_index_exclusive=4,
        midpoint_target_sec=1.0,
        midpoint_frame_index=2,
        midpoint_timestamp_sec=1.0,
        midpoint_error_ms=0.0,
        cameras=cameras,
    )


def test_frozen_quantiles_for_k1():
    assert interior_quantile_fractions(1) == [0.5]


def test_frozen_quantiles_for_k2():
    assert interior_quantile_fractions(2) == [
        1 / 3,
        2 / 3,
    ]


def test_frozen_quantiles_for_k3():
    assert interior_quantile_fractions(3) == [
        0.25,
        0.5,
        0.75,
    ]


def test_unsupported_k_is_rejected():
    with pytest.raises(ValueError):
        interior_quantile_fractions(4)


def test_k1_reproduces_midpoint_selection():
    recs = records([0.0, 0.5, 1.0, 1.5])
    item = build_multi_keyframe_selection(
        records=recs,
        temporal_slice=temporal_slice(),
        k=1,
    )
    assert item.keyframes[0].selected_frame_index == 2
    assert item.keyframes[0].selected_timestamp_sec == 1.0


def test_nearest_sample_tie_breaks_to_lower_frame():
    recs = records([0.0, 0.8, 1.2, 1.8])
    item = build_multi_keyframe_selection(
        records=recs,
        temporal_slice=TemporalSlice(
            **{
                **temporal_slice().model_dump(),
                "midpoint_frame_index": 1,
                "midpoint_timestamp_sec": 0.8,
                "midpoint_error_ms": 200.0,
                "cameras": [
                    {
                        **camera,
                        "frame_index": 1,
                        "timestamp_sec": 0.8,
                    }
                    for camera in temporal_slice().model_dump()["cameras"]
                ],
            }
        ),
        k=1,
    )
    assert item.keyframes[0].selected_frame_index == 1


def test_k3_preserves_unique_original_camera_evidence():
    recs = records([0.0, 0.5, 1.0, 1.5])
    item = build_multi_keyframe_selection(
        records=recs,
        temporal_slice=temporal_slice(),
        k=3,
    )
    assert [
        keyframe.selected_frame_index
        for keyframe in item.keyframes
    ] == [1, 2, 3]
    assert item.shared_sample_budget == 3
    assert item.evidence_image_budget == 6
    for keyframe in item.keyframes:
        assert [camera.camera for camera in keyframe.cameras] == [
            "front",
            "wrist",
        ]
        assert all(
            camera.image_relpath.endswith(
                f"{keyframe.selected_frame_index:06d}.jpg"
            )
            for camera in keyframe.cameras
        )
