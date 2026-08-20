from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidencemm.temporal_evidence import FrameRecord
from evidencemm.temporal_eval import (
    TemporalEventAnnotation,
    evaluate_event_coverage,
)
from evidencemm.temporal_slicing import (
    CameraMidpointEvidence,
    TemporalSlice,
)


def frame(
    camera: str,
    index: int,
    timestamp: float,
) -> FrameRecord:
    return FrameRecord(
        episode_id="ep0",
        camera=camera,
        frame_index=index,
        timestamp_sec=timestamp,
        sample_timestamp_ns=1000 + index,
        source_timestamp_ns=900 + index,
        source_age_ms=5.0,
        camera_sequence=index,
        image_relpath=(
            f"{camera}/{index:06d}.jpg"
        ),
        image_sha256=(
            ("1" if camera == "front" else "2")
            * 64
        ),
        width_px=640,
        height_px=480,
    )


def slice_item(
    index: int,
    midpoint_frame: int,
    midpoint_sec: float,
) -> TemporalSlice:
    cameras = [
        CameraMidpointEvidence(
            camera=camera,
            frame_index=midpoint_frame,
            timestamp_sec=midpoint_sec,
            source_timestamp_ns=1000 + midpoint_frame,
            source_age_ms=5.0,
            camera_sequence=midpoint_frame,
            image_relpath=(
                f"{camera}/{midpoint_frame:06d}.jpg"
            ),
            image_sha256=(
                ("1" if camera == "front" else "2")
                * 64
            ),
            width_px=640,
            height_px=480,
        )
        for camera in ("front", "wrist")
    ]

    return TemporalSlice(
        slice_group_id=f"ep0_slice_{index:04d}",
        episode_id="ep0",
        start_sec=index * 2.0,
        end_sec=(index + 1) * 2.0,
        start_frame_index=index * 2,
        end_frame_index_exclusive=index * 2 + 2,
        midpoint_target_sec=index * 2.0 + 1.0,
        midpoint_frame_index=midpoint_frame,
        midpoint_timestamp_sec=midpoint_sec,
        midpoint_error_ms=0.0,
        cameras=cameras,
    )


def test_verified_annotation_requires_bounds():
    with pytest.raises(ValidationError):
        TemporalEventAnnotation(
            event_id="e0",
            episode_id="ep0",
            event_type="grasp",
            description="event",
            status="verified",
            evidence_cameras=["front"],
        )


def test_event_coverage_hit():
    records = []
    for index, timestamp in enumerate(
        [0.0, 1.0, 2.0, 3.0]
    ):
        for camera in ("front", "wrist"):
            records.append(
                frame(
                    camera,
                    index,
                    timestamp,
                )
            )

    annotation = TemporalEventAnnotation(
        event_id="e0",
        episode_id="ep0",
        event_type="grasp",
        description="event",
        status="verified",
        start_frame_index=1,
        end_frame_index_inclusive=1,
        evidence_cameras=[
            "front",
            "wrist",
        ],
    )

    result = evaluate_event_coverage(
        annotations=[annotation],
        records=records,
        slices=[
            slice_item(0, 1, 1.0),
            slice_item(1, 3, 3.0),
        ],
    )

    assert result["event_coverage"] == 1.0
    assert result["events"][0]["covered"] is True


def test_event_coverage_miss_is_preserved():
    records = []
    for index, timestamp in enumerate(
        [0.0, 1.0, 2.0, 3.0]
    ):
        for camera in ("front", "wrist"):
            records.append(
                frame(
                    camera,
                    index,
                    timestamp,
                )
            )

    annotation = TemporalEventAnnotation(
        event_id="e0",
        episode_id="ep0",
        event_type="short_event",
        description="event",
        status="verified",
        start_frame_index=2,
        end_frame_index_inclusive=2,
        evidence_cameras=["front"],
    )

    result = evaluate_event_coverage(
        annotations=[annotation],
        records=records,
        slices=[
            slice_item(0, 1, 1.0),
            slice_item(1, 3, 3.0),
        ],
    )

    assert result["event_coverage"] == 0.0
    assert result["events"][0]["covered"] is False
