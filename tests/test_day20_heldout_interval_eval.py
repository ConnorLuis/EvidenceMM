import json
from dataclasses import dataclass

import pytest

from evidencemm.heldout_interval_eval import (
    HeldoutGoldEvent,
    aggregate_metrics,
    build_report,
    evaluate_verified_event,
    interval_iou,
    load_heldout_gold_events,
    validate_expected_heldout_gold,
    validate_frozen_day19_model,
)


@dataclass(frozen=True)
class Proposal:
    center_frame: int
    start_frame: int
    end_frame: int
    radius_frames: int
    center_timestamp_sec: float
    reasons: tuple[str, ...]


def test_interval_iou_exact():
    assert interval_iou(
        10, 12, 10, 12
    ) == pytest.approx(1.0)


def test_interval_iou_partial():
    assert interval_iou(
        10, 14, 12, 16
    ) == pytest.approx(3 / 7)


def test_interval_iou_disjoint():
    assert interval_iou(
        0, 2, 3, 5
    ) == 0.0


def test_load_heldout_gold_skips_nonheldout_before_interval_access(
    tmp_path,
):
    path = tmp_path / "gt.jsonl"
    rows = [
        {
            "event_id": "development_bad",
            "episode_id": "development",
            "observed_failure_mode": "grasp_drop",
            "review_disposition": "verified",
            "failure_interval": "this would be invalid if accessed",
        },
        {
            "event_id": "held_event",
            "episode_id": "held",
            "observed_failure_mode": "grasp_drop",
            "review_disposition": "verified",
            "failure_interval": {
                "start_frame": 10,
                "end_frame": 12,
            },
        },
    ]
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    events = load_heldout_gold_events(
        path,
        allowed_episode_ids={"held"},
    )
    assert [
        event.event_id
        for event in events
    ] == ["held_event"]


def test_validate_expected_heldout_gold():
    events = [
        HeldoutGoldEvent(
            event_id="e1",
            episode_id="p1",
            observed_failure_mode="grasp_drop",
            review_disposition="verified",
            start_frame=10,
            end_frame=12,
        ),
        HeldoutGoldEvent(
            event_id="e2",
            episode_id="p2",
            observed_failure_mode="post_place_collision",
            review_disposition="verified",
            start_frame=20,
            end_frame=25,
        ),
    ]
    validate_expected_heldout_gold(
        events,
        expected_event_count=2,
        expected_verified_count=2,
        expected_unresolved_count=0,
        expected_event_ids=["e1", "e2"],
    )


def test_evaluate_verified_event_selects_highest_iou():
    event = HeldoutGoldEvent(
        event_id="e",
        episode_id="p",
        observed_failure_mode="grasp_drop",
        review_disposition="verified",
        start_frame=10,
        end_frame=12,
    )
    proposals = [
        Proposal(
            center_frame=5,
            start_frame=3,
            end_frame=7,
            radius_frames=2,
            center_timestamp_sec=0.3,
            reasons=("front_visual_motion",),
        ),
        Proposal(
            center_frame=11,
            start_frame=9,
            end_frame=13,
            radius_frames=2,
            center_timestamp_sec=0.7,
            reasons=("wrist_visual_motion",),
        ),
    ]
    result = evaluate_verified_event(
        event=event,
        proposals=proposals,
    )
    assert result.best_proposal_center_frame == 11
    assert result.best_iou == pytest.approx(3 / 5)
    assert result.overlap_hit is True


def test_aggregate_metrics_reports_threshold_recall():
    event = HeldoutGoldEvent(
        event_id="e",
        episode_id="p",
        observed_failure_mode="grasp_drop",
        review_disposition="verified",
        start_frame=10,
        end_frame=12,
    )
    result = evaluate_verified_event(
        event=event,
        proposals=[
            Proposal(
                center_frame=11,
                start_frame=9,
                end_frame=13,
                radius_frames=2,
                center_timestamp_sec=0.7,
                reasons=("wrist_visual_motion",),
            )
        ],
    )
    metrics = aggregate_metrics(
        [result],
        iou_thresholds=[0.1, 0.25, 0.5],
    )
    assert metrics["overall"]["event_recall"] == 1.0
    assert metrics["overall"]["recall_at_iou"] == {
        "0.10": 1.0,
        "0.25": 1.0,
        "0.50": 1.0,
    }


def test_validate_frozen_day19_model_accepts_exact_contract():
    model = {
        "schema_version": (
            "evidencemm_day19_interval_localizer_model_v1"
        ),
        "model_status": (
            "development_selected_interval_proposal_localizer"
        ),
        "model_selection_split": "development",
        "provenance": {
            "benchmark_split_sha256": "a",
            "frozen_selector_config_sha256": "b",
            "human_gt_sha256": "c",
        },
        "anti_leakage": {
            "held_out_gt_used_for_model_selection": False,
            "held_out_metrics_reported": False,
        },
        "localizer": {
            "selected_radius_frames": 5,
        },
    }
    assert validate_frozen_day19_model(
        model,
        expected_selected_radius_frames=5,
        split_sha256="a",
        selector_config_sha256="b",
        human_gt_sha256="c",
    ) == 5


def test_validate_frozen_day19_model_rejects_radius_change():
    model = {
        "schema_version": (
            "evidencemm_day19_interval_localizer_model_v1"
        ),
        "model_status": (
            "development_selected_interval_proposal_localizer"
        ),
        "model_selection_split": "development",
        "provenance": {
            "benchmark_split_sha256": "a",
            "frozen_selector_config_sha256": "b",
            "human_gt_sha256": "c",
        },
        "anti_leakage": {
            "held_out_gt_used_for_model_selection": False,
            "held_out_metrics_reported": False,
        },
        "localizer": {
            "selected_radius_frames": 3,
        },
    }
    with pytest.raises(
        ValueError,
        match="selected radius",
    ):
        validate_frozen_day19_model(
            model,
            expected_selected_radius_frames=5,
            split_sha256="a",
            selector_config_sha256="b",
            human_gt_sha256="c",
        )


def test_build_report_seals_heldout_against_future_model_selection():
    event = HeldoutGoldEvent(
        event_id="e",
        episode_id="p",
        observed_failure_mode="grasp_drop",
        review_disposition="verified",
        start_frame=10,
        end_frame=12,
    )
    proposals = {
        "p": [
            Proposal(
                center_frame=11,
                start_frame=6,
                end_frame=16,
                radius_frames=5,
                center_timestamp_sec=0.7,
                reasons=("wrist_visual_motion",),
            )
        ]
    }
    report = build_report(
        frozen_after_day19_commit="1" * 40,
        day19_model_blob_sha1="2" * 40,
        split_sha256="3" * 64,
        model_sha256="4" * 64,
        selector_config_sha256="5" * 64,
        human_gt_sha256="6" * 64,
        selected_radius_frames=5,
        proposals_by_episode=proposals,
        gold_events=[event],
        iou_thresholds=[0.1, 0.25, 0.5],
    )
    assert report["anti_leakage"][
        "model_selection_performed"
    ] is False
    assert report["anti_leakage"][
        "radius_tuned_on_held_out"
    ] is False
    assert report["evaluation_seal"][
        "same_held_out_set_may_be_used_for_future_model_selection"
    ] is False
