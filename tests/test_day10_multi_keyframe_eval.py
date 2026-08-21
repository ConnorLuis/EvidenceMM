from __future__ import annotations

from evidencemm.multi_keyframe_eval import (
    evaluate_multi_keyframe_event_coverage,
)
from evidencemm.multi_keyframe_selection import (
    CameraKeyframeEvidence,
    MultiKeyframeSelection,
    TemporalKeyframe,
)
from evidencemm.temporal_eval import TemporalEventAnnotation

from test_day7_temporal_slicing import records


def camera(
    name: str,
    frame_index: int,
    timestamp_sec: float,
) -> CameraKeyframeEvidence:
    return CameraKeyframeEvidence(
        camera=name,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        source_timestamp_ns=1000 + frame_index,
        source_age_ms=1.0,
        camera_sequence=frame_index,
        image_relpath=f"{name}/{frame_index:06d}.jpg",
        image_sha256=(
            ("1" if name == "front" else "2")
            * 64
        ),
        width_px=640,
        height_px=480,
    )


def keyframe(
    *,
    rank: int,
    fraction: float,
    frame_index: int,
    timestamp_sec: float,
) -> TemporalKeyframe:
    return TemporalKeyframe(
        keyframe_rank=rank,
        target_fraction=fraction,
        target_timestamp_sec=timestamp_sec,
        selected_frame_index=frame_index,
        selected_timestamp_sec=timestamp_sec,
        selection_error_ms=0.0,
        cameras=[
            camera("front", frame_index, timestamp_sec),
            camera("wrist", frame_index, timestamp_sec),
        ],
    )


def selection(
    *,
    slice_id: str,
    k: int,
    frames: list[tuple[int, float]],
) -> MultiKeyframeSelection:
    fractions = {
        1: [0.5],
        2: [1 / 3, 2 / 3],
        3: [0.25, 0.5, 0.75],
    }[k]
    return MultiKeyframeSelection(
        slice_group_id=slice_id,
        episode_id="ep0",
        start_sec=0.0,
        end_sec=10.0,
        start_frame_index=0,
        end_frame_index_exclusive=10,
        k=k,
        shared_sample_budget=k,
        evidence_image_budget=k * 2,
        keyframes=[
            keyframe(
                rank=index + 1,
                fraction=fractions[index],
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
            for index, (frame_index, timestamp_sec)
            in enumerate(frames)
        ],
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


def test_multi_keyframe_event_is_hit_by_any_selected_frame():
    recs = records([0.0, 1.0, 2.0, 3.0])
    result = evaluate_multi_keyframe_event_coverage(
        annotations=[annotation(1, 1)],
        records=recs,
        selections=[
            selection(
                slice_id="slice0",
                k=2,
                frames=[
                    (0, 0.0),
                    (1, 1.0),
                ],
            )
        ],
    )
    assert result.covered_events == 1
    assert result.event_coverage == 1.0
    assert result.shared_sample_budget == 2
    assert result.evidence_image_budget == 4


def test_multi_keyframe_nearest_distance_uses_all_keyframes():
    recs = records([0.0, 1.0, 2.0, 3.0])
    result = evaluate_multi_keyframe_event_coverage(
        annotations=[annotation(2, 2)],
        records=recs,
        selections=[
            selection(
                slice_id="slice0",
                k=2,
                frames=[
                    (0, 0.0),
                    (3, 3.0),
                ],
            )
        ],
    )
    row = result.events[0]
    assert row["covered"] is False
    assert row["closest_selected_frame"] == 3
    assert row["closest_selected_to_event_center_ms"] == 1000.0
