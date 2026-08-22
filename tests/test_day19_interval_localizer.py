from dataclasses import dataclass

import pytest

from evidencemm.interval_localizer import (
    DevelopmentGoldEvent,
    build_interval_proposals,
    build_model_artifact,
    interval_iou,
    selection_key,
    signal_candidates,
    validate_model_artifact_no_gold_boundaries,
)


@dataclass(frozen=True)
class Candidate:
    frame_index: int
    timestamp_sec: float
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def test_signal_candidates_excludes_uniform_only():
    values = [
        Candidate(
            frame_index=0,
            timestamp_sec=0.0,
            reasons=("uniform_anchor",),
            metrics={},
        ),
        Candidate(
            frame_index=5,
            timestamp_sec=0.3,
            reasons=("uniform_anchor", "front_visual_motion"),
            metrics={"front_visual_motion": 0.2},
        ),
        Candidate(
            frame_index=9,
            timestamp_sec=0.6,
            reasons=("state_action_change",),
            metrics={"fused_state_action_score": 1.0},
        ),
    ]
    assert [
        item.frame_index
        for item in signal_candidates(values)
    ] == [5, 9]


def test_build_interval_proposals_clips_episode_edges():
    candidates = [
        Candidate(
            frame_index=1,
            timestamp_sec=0.1,
            reasons=("front_visual_motion",),
            metrics={},
        ),
        Candidate(
            frame_index=9,
            timestamp_sec=0.9,
            reasons=("wrist_visual_motion",),
            metrics={},
        ),
    ]
    proposals = build_interval_proposals(
        candidates,
        frame_count=10,
        radius_frames=3,
    )
    assert (
        proposals[0].start_frame,
        proposals[0].end_frame,
    ) == (0, 4)
    assert (
        proposals[1].start_frame,
        proposals[1].end_frame,
    ) == (6, 9)


def test_build_interval_proposals_rejects_negative_radius():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        build_interval_proposals(
            [
                Candidate(
                    frame_index=1,
                    timestamp_sec=0.1,
                    reasons=("front_visual_motion",),
                    metrics={},
                )
            ],
            frame_count=10,
            radius_frames=-1,
        )


@pytest.mark.parametrize(
    "a,b,c,d,expected",
    [
        (0, 4, 2, 6, 3 / 7),
        (0, 1, 2, 3, 0.0),
        (5, 5, 5, 5, 1.0),
    ],
)
def test_interval_iou(
    a,
    b,
    c,
    d,
    expected,
):
    assert interval_iou(
        a,
        b,
        c,
        d,
    ) == pytest.approx(expected)


def test_selection_key_prioritizes_event_recall_before_iou():
    high_recall = {
        "event_recall": 1.0,
        "recall_at_iou": {"0.25": 0.2},
        "mean_best_iou": 0.2,
        "radius_frames": 3,
    }
    lower_recall = {
        "event_recall": 0.8,
        "recall_at_iou": {"0.25": 1.0},
        "mean_best_iou": 0.9,
        "radius_frames": 1,
    }
    assert selection_key(
        high_recall
    ) > selection_key(
        lower_recall
    )


def test_selection_key_uses_smaller_radius_as_final_tie_break():
    small = {
        "event_recall": 1.0,
        "recall_at_iou": {"0.25": 0.8},
        "mean_best_iou": 0.4,
        "radius_frames": 3,
    }
    large = {
        "event_recall": 1.0,
        "recall_at_iou": {"0.25": 0.8},
        "mean_best_iou": 0.4,
        "radius_frames": 4,
    }
    assert selection_key(small) > selection_key(large)


def test_model_artifact_contains_no_gold_boundaries():
    grid_metrics = [
        {
            "event_count": 5,
            "event_recall": 1.0,
            "mean_best_iou": 0.4,
            "median_best_iou": 0.4,
            "recall_at_iou": {
                "0.10": 1.0,
                "0.25": 0.8,
                "0.50": 0.4,
            },
            "mean_onset_abs_error_frames": 2.0,
            "mean_offset_abs_error_frames": 2.0,
            "radius_frames": 3,
            "episode_count": 6,
            "mean_proposals_per_episode": 17.0,
            "min_proposals_per_episode": 15,
            "max_proposals_per_episode": 18,
        }
    ]
    artifact = build_model_artifact(
        split_sha256="1" * 64,
        selector_config_sha256="2" * 64,
        human_gt_sha256="3" * 64,
        frozen_after_day18_commit="4" * 40,
        development_episode_count=54,
        development_anomaly_episode_count=6,
        held_out_episode_count=14,
        held_out_anomaly_episode_count=2,
        development_verified_event_count=5,
        development_reviewed_unresolved_count=2,
        radius_grid_frames=[3],
        iou_thresholds=[0.1, 0.25, 0.5],
        selected_radius_frames=3,
        grid_metrics=grid_metrics,
    )
    validate_model_artifact_no_gold_boundaries(
        artifact
    )
    serialized = str(artifact)
    assert "failure_interval" not in serialized
    assert "start_frame" not in serialized
    assert "end_frame" not in serialized
    assert (
        artifact["anti_leakage"][
            "held_out_gt_used_for_model_selection"
        ]
        is False
    )


def test_development_gold_event_verified_property():
    event = DevelopmentGoldEvent(
        event_id="e",
        episode_id="p",
        observed_failure_mode="grasp_drop",
        review_disposition="verified",
        start_frame=10,
        end_frame=12,
    )
    assert event.is_verified is True
