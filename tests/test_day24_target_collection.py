from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evidencemm.day24_target_collection import (
    TARGET_DIRECTION,
    TARGET_MAGNITUDE_MM,
    TARGET_REPEAT_GROUPS,
    analyze_collection,
    build_record,
    evaluate_experimental_validity,
    expected_day24_rows_from_day22,
    validate_day24_plan_shape,
)


def synthetic_day22():
    rows = []
    rotation = {
        1: "target_offset_or_perception",
        2: "gripper_close_timing",
        0: "trajectory_execution_deviation",
    }
    intervention = {
        "target_offset_or_perception": "object_target_pose_offset",
        "gripper_close_timing": "manual_gripper_close_timing_shift",
        "trajectory_execution_deviation": "manual_bounded_trajectory_deviation",
    }
    for g in range(1, 16):
        group = f"rcv2_g{g:02d}"
        causes = {
            1: ("clean_control", "none_clean", "none", "False"),
            2: ("controlled_cause", "target_offset_or_perception", "object_target_pose_offset", "False"),
            3: ("controlled_cause", "gripper_close_timing", "manual_gripper_close_timing_shift", "False"),
            4: ("controlled_cause", "trajectory_execution_deviation", "manual_bounded_trajectory_deviation", "False"),
            6: ("insufficient_evidence_candidate", "unknown", "ambiguity_protocol", "False"),
        }
        repeat_cause = rotation[g % 3]
        causes[5] = ("controlled_cause", repeat_cause, intervention[repeat_cause], "True")
        for slot in range(1, 7):
            role, cause, iv, repeat = causes[slot]
            rows.append({
                "plan_row_id": f"{group}_s{slot:02d}",
                "pair_group_id": group,
                "slot_index": str(slot),
                "slot_role": role,
                "planned_physical_cause": cause,
                "planned_intervention_type": iv,
                "repeat_slot": repeat,
            })
    return rows


def test_day24_derived_plan_shape():
    plan = expected_day24_rows_from_day22(synthetic_day22())
    validate_day24_plan_shape(plan)
    assert len(plan) == 35
    assert sum(r["slot_role"] == "clean_control" for r in plan) == 15
    targets = [r for r in plan if r["planned_physical_cause"] == "target_offset_or_perception"]
    assert len(targets) == 20
    repeats = {r["pair_group_id"] for r in targets if r["repeat_slot"] == "true"}
    assert repeats == set(TARGET_REPEAT_GROUPS)


def test_target_parameter_is_frozen_medium_pilot_value():
    plan = expected_day24_rows_from_day22(synthetic_day22())
    targets = [r for r in plan if r["planned_physical_cause"] == "target_offset_or_perception"]
    assert {r["parameter_direction"] for r in targets} == {TARGET_DIRECTION}
    assert {float(r["parameter_value"]) for r in targets} == {TARGET_MAGNITUDE_MM}
    assert {r["parameter_unit"] for r in targets} == {"mm"}


@pytest.mark.parametrize(
    "role,success,applied,valid,reason",
    [
        ("clean", True, False, True, ""),
        ("clean", False, False, False, "clean_task_failure"),
        ("target", False, True, True, ""),
        ("target", True, True, False, "target_intervention_did_not_induce_failure"),
    ],
)
def test_experimental_eligibility(role, success, applied, valid, reason):
    row = {
        "slot_role": "clean_control" if role == "clean" else "controlled_cause",
        "planned_physical_cause": "none_clean" if role == "clean" else "target_offset_or_perception",
    }
    actual, actual_reason = evaluate_experimental_validity(
        plan_row=row,
        technical_valid=True,
        task_success=success,
        intervention_applied=applied,
        single_primary_intervention=True,
        changed_factor_observable=True,
        safety_abort=False,
        hardware_fault=False,
    )
    assert actual is valid
    assert actual_reason == reason


def test_no_future_split_in_analysis():
    plan = expected_day24_rows_from_day22(synthetic_day22())
    analysis = analyze_collection(plan_rows=plan, records=[], raw_root=None)
    assert analysis["future_split_materialized"] is False
    assert analysis["expected_canonical_episode_count"] == 35
