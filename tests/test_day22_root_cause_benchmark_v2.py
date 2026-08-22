from collections import Counter

import pytest
from pydantic import ValidationError

from evidencemm.root_cause_benchmark_v2 import (
    CAUSES,
    DiagnosticDecisionGT,
    EvidenceAnswerabilityGT,
    FailureIntervalV2,
    PhysicalCauseGT,
    RootCauseReviewRecordV2,
    build_collection_plan,
    collection_plan_csv_bytes,
    future_split_rank,
    load_collection_plan_csv,
    materialize_future_group_split,
    repeat_cause_for_group,
    validate_collection_plan,
)


def test_collection_plan_has_90_rows():
    rows = build_collection_plan()
    assert len(rows) == 90


def test_collection_plan_has_15_complete_pair_groups():
    rows = build_collection_plan()
    counts = Counter(
        row.pair_group_id
        for row in rows
    )
    assert len(counts) == 15
    assert set(counts.values()) == {6}


def test_each_cause_has_20_final_slots():
    rows = build_collection_plan()
    counts = Counter(
        row.planned_physical_cause
        for row in rows
        if row.slot_role == "controlled_cause"
    )
    assert counts == Counter(
        {
            "target_offset_or_perception": 20,
            "gripper_close_timing": 20,
            "trajectory_execution_deviation": 20,
        }
    )


def test_repeat_cause_rotation_is_balanced():
    repeats = Counter(
        repeat_cause_for_group(
            index
        )
        for index in range(
            1,
            16,
        )
    )
    assert repeats == Counter(
        {
            "target_offset_or_perception": 5,
            "gripper_close_timing": 5,
            "trajectory_execution_deviation": 5,
        }
    )


def test_collection_plan_csv_round_trip():
    rows = build_collection_plan()
    payload = collection_plan_csv_bytes(
        rows
    )
    loaded = load_collection_plan_csv(
        payload.decode(
            "utf-8"
        )
    )
    assert loaded == rows


def test_collection_plan_rejects_missing_row():
    rows = build_collection_plan()
    with pytest.raises(
        ValueError,
        match="expected 90",
    ):
        validate_collection_plan(
            rows[:-1]
        )


def test_future_split_is_pair_group_atomic_and_10_5():
    group_ids = [
        f"rcv2_g{index:02d}"
        for index in range(
            1,
            16,
        )
    ]
    development, held_out = (
        materialize_future_group_split(
            group_ids,
            seed=(
                "evidencemm-root-cause-v2-split-v3"
            ),
            held_out_group_count=5,
        )
    )
    assert len(development) == 10
    assert len(held_out) == 5
    assert set(development).isdisjoint(
        held_out
    )
    assert set(
        development
    ) | set(
        held_out
    ) == set(group_ids)


def test_future_split_rank_is_deterministic():
    a = future_split_rank(
        "rcv2_g01",
        seed="seed",
    )
    b = future_split_rank(
        "rcv2_g01",
        seed="seed",
    )
    assert a == b
    assert len(a) == 64


def test_verified_answerable_cause_requires_robot_and_manual_support():
    interval = FailureIntervalV2(
        start_frame=10,
        end_frame=20,
        start_sec=1.0,
        end_sec=2.0,
    )
    with pytest.raises(
        ValidationError,
        match="supporting robot refs",
    ):
        RootCauseReviewRecordV2(
            episode_id="episode",
            pair_group_id="rcv2_g01",
            technical_valid=True,
            experimental_valid=True,
            task_success=False,
            intervention_verified=True,
            physical_cause_gt=(
                PhysicalCauseGT.GRIPPER_CLOSE_TIMING
            ),
            evidence_answerability_gt=(
                EvidenceAnswerabilityGT.ANSWERABLE
            ),
            diagnostic_decision_gt=(
                DiagnosticDecisionGT.GRIPPER_CLOSE_TIMING
            ),
            failure_interval=interval,
            supporting_robot_refs=[],
            supporting_manual_refs=[
                {"page_number": 2}
            ],
            confidence=0.9,
            review_notes="verified",
        )


def test_verified_answerable_cause_accepts_matching_contract():
    interval = FailureIntervalV2(
        start_frame=10,
        end_frame=20,
        start_sec=1.0,
        end_sec=2.0,
    )
    record = RootCauseReviewRecordV2(
        episode_id="episode",
        pair_group_id="rcv2_g01",
        technical_valid=True,
        experimental_valid=True,
        task_success=False,
        intervention_verified=True,
        physical_cause_gt=(
            PhysicalCauseGT.TRAJECTORY_EXECUTION_DEVIATION
        ),
        evidence_answerability_gt=(
            EvidenceAnswerabilityGT.ANSWERABLE
        ),
        diagnostic_decision_gt=(
            DiagnosticDecisionGT.TRAJECTORY_EXECUTION_DEVIATION
        ),
        failure_interval=interval,
        supporting_robot_refs=[
            {"frame_index": 15}
        ],
        supporting_manual_refs=[
            {"page_number": 4}
        ],
        confidence=0.9,
        review_notes="single controlled intervention verified",
    )
    assert (
        record.diagnostic_decision_gt.value
        == "trajectory_execution_deviation"
    )


def test_insufficient_evidence_is_decision_not_required_physical_cause():
    interval = FailureIntervalV2(
        start_frame=10,
        end_frame=20,
        start_sec=1.0,
        end_sec=2.0,
    )
    record = RootCauseReviewRecordV2(
        episode_id="episode",
        pair_group_id="rcv2_g01",
        technical_valid=True,
        experimental_valid=True,
        task_success=False,
        intervention_verified=True,
        physical_cause_gt=(
            PhysicalCauseGT.TARGET_OFFSET_OR_PERCEPTION
        ),
        evidence_answerability_gt=(
            EvidenceAnswerabilityGT.INSUFFICIENT_EVIDENCE
        ),
        diagnostic_decision_gt=(
            DiagnosticDecisionGT.INSUFFICIENT_EVIDENCE
        ),
        failure_interval=interval,
        confidence=0.7,
        review_notes=(
            "cause known from admin intervention but not "
            "supported by model-visible evidence"
        ),
    )
    assert (
        record.physical_cause_gt.value
        == "target_offset_or_perception"
    )
    assert (
        record.diagnostic_decision_gt.value
        == "insufficient_evidence"
    )


def test_clean_success_has_no_failure_interval():
    record = RootCauseReviewRecordV2(
        episode_id="episode",
        pair_group_id="rcv2_g01",
        technical_valid=True,
        experimental_valid=True,
        task_success=True,
        intervention_verified=False,
        physical_cause_gt=(
            PhysicalCauseGT.NONE_CLEAN
        ),
        evidence_answerability_gt=(
            EvidenceAnswerabilityGT.NOT_APPLICABLE_CLEAN
        ),
        diagnostic_decision_gt=(
            DiagnosticDecisionGT.CLEAN_SUCCESS
        ),
        failure_interval=None,
        confidence=1.0,
        review_notes="matched clean control",
    )
    assert record.task_success is True


def test_answerable_decision_must_match_physical_cause():
    interval = FailureIntervalV2(
        start_frame=10,
        end_frame=20,
        start_sec=1.0,
        end_sec=2.0,
    )
    with pytest.raises(
        ValidationError,
        match="must equal physical_cause_gt",
    ):
        RootCauseReviewRecordV2(
            episode_id="episode",
            pair_group_id="rcv2_g01",
            technical_valid=True,
            experimental_valid=True,
            task_success=False,
            intervention_verified=True,
            physical_cause_gt=(
                PhysicalCauseGT.GRIPPER_CLOSE_TIMING
            ),
            evidence_answerability_gt=(
                EvidenceAnswerabilityGT.ANSWERABLE
            ),
            diagnostic_decision_gt=(
                DiagnosticDecisionGT.TARGET_OFFSET_OR_PERCEPTION
            ),
            failure_interval=interval,
            supporting_robot_refs=[
                {"frame_index": 15}
            ],
            supporting_manual_refs=[
                {"page_number": 2}
            ],
            confidence=0.9,
            review_notes="invalid mismatch",
        )
