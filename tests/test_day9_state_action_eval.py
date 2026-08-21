from __future__ import annotations

from evidencemm.state_action_eval import (
    evaluate_state_action_event_coverage,
)
from evidencemm.state_action_selection import (
    JointVector,
    StateActionAwareSelection,
)
from evidencemm.temporal_eval import TemporalEventAnnotation

from test_day7_temporal_slicing import records


def vector(value: float) -> JointVector:
    return JointVector(
        shoulder_pan=value,
        shoulder_lift=value,
        elbow_flex=value,
        wrist_flex=value,
        wrist_roll=value,
        gripper=value,
    )


def selection(
    *,
    slice_id: str,
    frame_index: int,
    timestamp_sec: float,
) -> StateActionAwareSelection:
    return StateActionAwareSelection(
        slice_group_id=slice_id,
        episode_id="ep0",
        start_sec=0.0,
        end_sec=10.0,
        start_frame_index=0,
        end_frame_index_exclusive=10,
        selected_frame_index=frame_index,
        selected_timestamp_sec=timestamp_sec,
        state_change_rms=0.2,
        action_change_rms=0.5,
        fused_state_action_score=0.5,
        tracking_gap_rms=1.0,
        observation=vector(1.0),
        action=vector(2.0),
        state_delta=vector(0.2),
        action_delta=vector(0.5),
        tracking_error=vector(1.0),
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


def test_state_action_event_coverage_hits_inside_gold():
    recs = records([0.0, 1.0, 2.0, 3.0])
    result = evaluate_state_action_event_coverage(
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


def test_state_action_event_coverage_reports_closest_frame():
    recs = records([0.0, 1.0, 2.0, 3.0])
    result = evaluate_state_action_event_coverage(
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
