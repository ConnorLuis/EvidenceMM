from __future__ import annotations

from evidencemm.motion_eval import evaluate_motion_event_coverage
from evidencemm.motion_selection import (
    CameraMotionEvidence,
    MotionAwareSelection,
)
from evidencemm.temporal_eval import TemporalEventAnnotation

from test_day7_temporal_slicing import records


def selection(
    *,
    slice_id: str,
    frame_index: int,
    timestamp_sec: float,
) -> MotionAwareSelection:
    cameras = []
    for camera in ("front", "wrist"):
        cameras.append(
            CameraMotionEvidence(
                camera=camera,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                source_timestamp_ns=1_000_000 + frame_index,
                source_age_ms=5.0,
                camera_sequence=frame_index,
                image_relpath=f"{camera}/{frame_index:06d}.jpg",
                image_sha256=(
                    ("1" if camera == "front" else "2") * 64
                ),
                motion_score=2.0,
            )
        )
    return MotionAwareSelection(
        slice_group_id=slice_id,
        episode_id="ep0",
        start_sec=0.0,
        end_sec=10.0,
        start_frame_index=0,
        end_frame_index_exclusive=10,
        selected_frame_index=frame_index,
        selected_timestamp_sec=timestamp_sec,
        front_motion_score=1.0,
        wrist_motion_score=2.0,
        fused_motion_score=2.0,
        cameras=cameras,
    )


def annotation(
    start: int,
    end: int,
) -> TemporalEventAnnotation:
    return TemporalEventAnnotation(
        event_id="event0",
        episode_id="ep0",
        event_type="test_event",
        description="test",
        status="verified",
        start_frame_index=start,
        end_frame_index_inclusive=end,
        evidence_cameras=["front", "wrist"],
    )


def test_motion_event_coverage_hits_selected_frame_inside_gold():
    recs = records([0.0, 1.0, 2.0, 3.0])
    result = evaluate_motion_event_coverage(
        annotations=[annotation(1, 2)],
        records=recs,
        selections=[
            selection(
                slice_id="slice0",
                frame_index=2,
                timestamp_sec=2.0,
            )
        ],
    )
    assert result.covered_events == 1
    assert result.event_coverage == 1.0
    assert result.shared_sample_budget == 1
    assert result.evidence_image_budget == 2


def test_motion_event_coverage_reports_closest_selected_frame():
    recs = records([0.0, 1.0, 2.0, 3.0])
    result = evaluate_motion_event_coverage(
        annotations=[annotation(1, 1)],
        records=recs,
        selections=[
            selection(
                slice_id="slice0",
                frame_index=0,
                timestamp_sec=0.0,
            ),
            selection(
                slice_id="slice1",
                frame_index=2,
                timestamp_sec=2.0,
            ),
        ],
    )
    row = result.events[0]
    assert row["covered"] is False
    assert row["closest_selected_frame"] == 0
    assert row["closest_selected_to_event_center_ms"] == 1000.0
