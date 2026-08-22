import pytest

from evidencemm.cross_domain_diagnostic_pack import (
    DAY20_REPORT_SCHEMA,
    DAY20_STATUS,
    DiagnosticCaseSeed,
    ROOT_CAUSE_DECISION,
    assert_no_review_labels_leaked,
    build_readiness,
    extract_diagnostic_case_seeds,
    representative_frames,
    validate_day20_report_contract,
    validate_manual_query_is_label_independent,
)


def _report():
    return {
        "schema_version": DAY20_REPORT_SCHEMA,
        "evaluation_status": DAY20_STATUS,
        "evaluation_split": "held_out",
        "anti_leakage": {
            "model_selection_performed": False,
            "radius_tuned_on_held_out": False,
            "post_heldout_tuning_allowed": False,
        },
        "evaluation_seal": {
            "frozen_model_final_evaluation": True,
            "same_held_out_set_may_be_used_for_future_model_selection": False,
        },
        "held_out_proposals": {
            "p1": [
                {
                    "start_frame": 10,
                    "center_frame": 15,
                    "end_frame": 20,
                }
            ],
            "p2": [
                {
                    "start_frame": 30,
                    "center_frame": 35,
                    "end_frame": 40,
                }
            ],
        },
        "event_results": [
            {
                "event_id": "e1",
                "episode_id": "p1",
                "best_proposal_start_frame": 10,
                "best_proposal_center_frame": 15,
                "best_proposal_end_frame": 20,
                "gold_start_frame": 12,
                "gold_end_frame": 18,
                "observed_failure_mode": "mode_a",
                "best_iou": 0.5,
            },
            {
                "event_id": "e2",
                "episode_id": "p2",
                "best_proposal_start_frame": 30,
                "best_proposal_center_frame": 35,
                "best_proposal_end_frame": 40,
                "gold_start_frame": 32,
                "gold_end_frame": 36,
                "observed_failure_mode": "mode_b",
                "best_iou": 0.4,
            },
        ],
    }


def test_validate_day20_report_contract():
    validate_day20_report_contract(
        _report()
    )


def test_validate_day20_report_rejects_tuning():
    report = _report()
    report["anti_leakage"][
        "radius_tuned_on_held_out"
    ] = True
    with pytest.raises(
        ValueError,
        match="radius_tuned_on_held_out",
    ):
        validate_day20_report_contract(
            report
        )


def test_extract_seeds_uses_matched_proposal_without_copying_gold():
    seeds = extract_diagnostic_case_seeds(
        _report(),
        expected_event_ids=[
            "e1",
            "e2",
        ],
        expected_episode_ids=[
            "p1",
            "p2",
        ],
    )
    assert seeds == [
        DiagnosticCaseSeed(
            event_id="e1",
            episode_id="p1",
            proposal_start_frame=10,
            proposal_center_frame=15,
            proposal_end_frame=20,
        ),
        DiagnosticCaseSeed(
            event_id="e2",
            episode_id="p2",
            proposal_start_frame=30,
            proposal_center_frame=35,
            proposal_end_frame=40,
        ),
    ]


def test_extract_seeds_rejects_untracked_best_proposal():
    report = _report()
    report["event_results"][0][
        "best_proposal_center_frame"
    ] = 16
    with pytest.raises(
        ValueError,
        match="not present",
    ):
        extract_diagnostic_case_seeds(
            report,
            expected_event_ids=[
                "e1",
                "e2",
            ],
            expected_episode_ids=[
                "p1",
                "p2",
            ],
        )


def test_representative_frames_are_start_center_end():
    seed = DiagnosticCaseSeed(
        event_id="e",
        episode_id="p",
        proposal_start_frame=10,
        proposal_center_frame=15,
        proposal_end_frame=20,
    )
    assert representative_frames(
        seed
    ) == (10, 15, 20)


def test_representative_frames_deduplicate():
    seed = DiagnosticCaseSeed(
        event_id="e",
        episode_id="p",
        proposal_start_frame=10,
        proposal_center_frame=10,
        proposal_end_frame=10,
    )
    assert representative_frames(
        seed
    ) == (10,)


def test_manual_query_rejects_review_label_leakage():
    with pytest.raises(
        ValueError,
        match="leaks reviewed failure labels",
    ):
        validate_manual_query_is_label_independent(
            "STS3215 grasp_drop troubleshooting"
        )


def test_manual_query_accepts_generic_component_query():
    validate_manual_query_is_label_independent(
        "STS3215 servo torque load voltage feedback"
    )


def test_readiness_forces_physical_root_cause_abstention():
    readiness = build_readiness(
        cross_domain_bundle_valid=True,
        document_item_count=3,
        robot_item_count=3,
        localization_origin=(
            "day20_gt_matched_best_proposal_for_post_eval_diagnostics"
        ),
        manual_support_status=(
            "retrieved_candidates_unlabeled_for_causal_support"
        ),
    )
    assert readiness[
        "root_cause_answerable"
    ] is False
    assert readiness[
        "decision"
    ] == ROOT_CAUSE_DECISION
    assert readiness["checks"][
        "manual_causal_ground_truth_available"
    ] is False


def test_review_label_leak_detector_rejects_gold_fields():
    with pytest.raises(
        ValueError,
        match="gold_start_frame",
    ):
        assert_no_review_labels_leaked(
            {
                "case": {
                    "gold_start_frame": 10,
                }
            }
        )


def test_review_label_leak_detector_accepts_predicted_proposal_fields():
    assert_no_review_labels_leaked(
        {
            "localized_proposal": {
                "start_frame": 10,
                "center_frame": 15,
                "end_frame": 20,
            }
        }
    )
