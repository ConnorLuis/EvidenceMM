import pytest

from evidencemm.failure_localization_eval import (
    CandidateFrame,
    GoldFailureEvent,
    aggregate_localization_results,
    evaluate_verified_event,
    interval_distance_frames,
    interval_distance_seconds,
)


def candidate(
    frame: int,
    timestamp: float,
) -> CandidateFrame:
    return CandidateFrame(
        frame_index=frame,
        timestamp_sec=timestamp,
        reasons=("test",),
        metrics={},
    )


def verified_event(
    *,
    mode: str = "grasp_drop",
) -> GoldFailureEvent:
    return GoldFailureEvent(
        event_id="ep_event_01",
        episode_id="ep",
        observed_failure_mode=mode,
        review_disposition="verified",
        start_frame=100,
        end_frame=110,
        start_sec=10.0,
        end_sec=11.0,
    )


def unresolved_event() -> GoldFailureEvent:
    return GoldFailureEvent(
        event_id="ep_event_02",
        episode_id="ep",
        observed_failure_mode=(
            "object_push_during_grasp"
        ),
        review_disposition=(
            "reviewed_unresolved"
        ),
        start_frame=None,
        end_frame=None,
        start_sec=None,
        end_sec=None,
    )


def test_interval_distance_frames():
    assert interval_distance_frames(
        105, 100, 110
    ) == 0
    assert interval_distance_frames(
        95, 100, 110
    ) == 5
    assert interval_distance_frames(
        114, 100, 110
    ) == 4


def test_interval_distance_seconds():
    assert interval_distance_seconds(
        10.5, 10.0, 11.0
    ) == 0.0
    assert interval_distance_seconds(
        9.75, 10.0, 11.0
    ) == 0.25
    assert interval_distance_seconds(
        11.2, 10.0, 11.0
    ) == pytest.approx(0.2)


def test_exact_event_hit():
    result = evaluate_verified_event(
        event=verified_event(),
        candidates=[
            candidate(80, 8.0),
            candidate(105, 10.5),
            candidate(130, 13.0),
        ],
        tolerance_frames=[5, 15],
    )

    assert result.exact_hit is True
    assert result.min_frame_distance == 0
    assert result.min_time_distance_ms == 0.0
    assert (
        result.candidate_frames_inside_interval
        == (105,)
    )


def test_tolerance_hit_without_exact_hit():
    result = evaluate_verified_event(
        event=verified_event(),
        candidates=[
            candidate(95, 9.5),
            candidate(130, 13.0),
        ],
        tolerance_frames=[5, 15],
    )

    assert result.exact_hit is False
    assert result.min_frame_distance == 5
    assert result.tolerance_hits == {
        "5": True,
        "15": True,
    }


def test_tolerance_miss_is_preserved():
    result = evaluate_verified_event(
        event=verified_event(),
        candidates=[
            candidate(80, 8.0),
            candidate(130, 13.0),
        ],
        tolerance_frames=[5, 15],
    )

    assert result.exact_hit is False
    assert result.min_frame_distance == 20
    assert result.tolerance_hits == {
        "5": False,
        "15": False,
    }


def test_aggregate_excludes_unresolved_from_denominator():
    exact = evaluate_verified_event(
        event=verified_event(
            mode="grasp_drop"
        ),
        candidates=[
            candidate(105, 10.5)
        ],
        tolerance_frames=[5, 15],
    )
    near = evaluate_verified_event(
        event=GoldFailureEvent(
            event_id="ep2_event_01",
            episode_id="ep2",
            observed_failure_mode=(
                "post_place_collision"
            ),
            review_disposition="verified",
            start_frame=200,
            end_frame=205,
            start_sec=20.0,
            end_sec=20.5,
        ),
        candidates=[
            candidate(210, 21.0)
        ],
        tolerance_frames=[5, 15],
    )

    metrics = aggregate_localization_results(
        results=[exact, near],
        unresolved_events=[
            unresolved_event()
        ],
        tolerance_frames=[5, 15],
    )

    assert metrics[
        "verified_event_count"
    ] == 2
    assert metrics[
        "reviewed_unresolved_count"
    ] == 1
    assert metrics["overall"][
        "exact_event_recall"
    ] == 0.5
    assert metrics["overall"][
        "tolerance_event_recall"
    ]["5"] == 1.0
    assert metrics[
        "reviewed_unresolved_event_ids"
    ] == ["ep_event_02"]
