from __future__ import annotations

from dataclasses import replace

import pytest

from evidencemm.day24_target_collection import TechnicalAudit
from evidencemm.day27_insufficient_evidence_collection import (
    AMBIGUITY_PROTOCOL,
    build_record,
    canonical_records,
    evaluate_experimental_validity,
    expected_day27_rows_from_day22,
    validate_day27_plan_shape,
    validate_preexisting_final_slots,
    analyze_collection,
    validate_final_analysis,
)


def day22_rows():
    rows = []
    for i in range(1, 16):
        group = f"rcv2_g{i:02d}"
        rows.append(
            {
                "plan_row_id": f"{group}_s06",
                "pair_group_id": group,
                "slot_index": "6",
                "slot_role": "insufficient_evidence_candidate",
                "planned_physical_cause": "unknown",
                "planned_intervention_type": "ambiguity_protocol",
                "repeat_slot": "false",
            }
        )
    return rows


def audit(valid=True):
    return TechnicalAudit(
        episode_id="20260824_999999",
        raw_episode_relpath="20260824_999999",
        recorder_script_version="episode_recorder_v7",
        technical_valid=valid,
        recorder_overall_pass=valid,
        failed_checks=[] if valid else ["x"],
        sample_count=900 if valid else 0,
        csv_row_count=900 if valid else 0,
        front_image_count=900 if valid else 0,
        wrist_image_count=900 if valid else 0,
        duration_seconds=59.93 if valid else None,
        max_tracking_error=10.0 if valid else None,
    )


def test_projection_is_exact_15_s06():
    rows = expected_day27_rows_from_day22(day22_rows())
    validate_day27_plan_shape(rows)
    assert len(rows) == 15
    assert rows[0]["plan_row_id"] == "rcv2_g01_s06"
    assert rows[-1]["plan_row_id"] == "rcv2_g15_s06"
    assert all(r["ambiguity_protocol"].startswith(AMBIGUITY_PROTOCOL + ":") for r in rows)


def test_projection_rejects_wrong_role():
    rows = day22_rows()
    rows[0]["slot_role"] = "clean_control"
    with pytest.raises(ValueError):
        expected_day27_rows_from_day22(rows)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("task_success", True, "ambiguity_candidate_task_success"),
        ("ambiguity_protocol_followed", False, "ambiguity_protocol_not_followed"),
        (
            "deliberate_known_cause_intervention",
            False,
            "assigned_ambiguity_challenge_not_applied",
        ),
        (
            "intentional_failure_injection",
            False,
            "predeclared_failure_challenge_not_applied",
        ),
        ("multiple_primary_interventions", True, "multiple_primary_interventions"),
        ("scene_comparable", False, "pair_group_setup_not_comparable"),
        ("safety_abort", True, "safety_abort"),
        ("hardware_fault", True, "hardware_fault"),
    ],
)
def test_invalid_conditions_require_recollection(field, value, reason):
    plan = expected_day27_rows_from_day22(day22_rows())[0]
    kwargs = dict(
        plan_row=plan,
        technical_valid=True,
        task_success=False,
        ambiguity_protocol_followed=True,
        deliberate_known_cause_intervention=True,
        intentional_failure_injection=True,
        multiple_primary_interventions=False,
        scene_comparable=True,
        safety_abort=False,
        hardware_fault=False,
    )
    kwargs[field] = value
    valid, got = evaluate_experimental_validity(**kwargs)
    assert valid is False
    assert got == reason


def test_technical_corruption_is_exclusion_not_insufficient():
    plan = expected_day27_rows_from_day22(day22_rows())[0]
    valid, reason = evaluate_experimental_validity(
        plan_row=plan,
        technical_valid=False,
        task_success=False,
        ambiguity_protocol_followed=True,
        deliberate_known_cause_intervention=True,
        intentional_failure_injection=True,
        multiple_primary_interventions=False,
        scene_comparable=True,
        safety_abort=False,
        hardware_fault=False,
    )
    assert not valid
    assert reason == "technical_exclusion"


def test_predeclared_single_challenge_failure_is_canonical_candidate():
    plan = expected_day27_rows_from_day22(day22_rows())[0]
    record = build_record(
        plan_row=plan,
        audit=audit(True),
        attempt_index=1,
        task_success=False,
        ambiguity_protocol_followed=True,
        deliberate_known_cause_intervention=True,
        intentional_failure_injection=True,
        multiple_primary_interventions=False,
        scene_comparable=True,
        safety_abort=False,
        hardware_fault=False,
        selected_canonical=True,
        operator_notes="v2 assigned challenge failure",
    )
    assert record["technical_valid"] == "true"
    assert record["experimental_valid"] == "true"
    assert record["selected_canonical"] == "true"
    assert record["exclusion_reason"] == ""


def test_success_is_retained_but_not_canonical():
    plan = expected_day27_rows_from_day22(day22_rows())[0]
    record = build_record(
        plan_row=plan,
        audit=audit(True),
        attempt_index=1,
        task_success=True,
        ambiguity_protocol_followed=True,
        deliberate_known_cause_intervention=True,
        intentional_failure_injection=True,
        multiple_primary_interventions=False,
        scene_comparable=True,
        safety_abort=False,
        hardware_fault=False,
        selected_canonical=True,
        operator_notes="v2 challenge task success",
    )
    assert record["technical_valid"] == "true"
    assert record["experimental_valid"] == "false"
    assert record["selected_canonical"] == "false"
    assert record["exclusion_reason"] == "ambiguity_candidate_task_success"


def test_duplicate_canonical_for_same_slot_is_rejected():
    plan = expected_day27_rows_from_day22(day22_rows())
    base = {
        "plan_row_id": "rcv2_g01_s06",
        "selected_canonical": "true",
    }
    with pytest.raises(ValueError):
        canonical_records(plan, [base, dict(base)])


def make_preexisting():
    day24_plan = []
    day24_records = []
    day25_plan = []
    day25_records = []
    day26_plan = []
    day26_records = []

    target_repeat_groups = {"rcv2_g01", "rcv2_g04", "rcv2_g07", "rcv2_g10", "rcv2_g13"}
    gripper_repeat_groups = {"rcv2_g02", "rcv2_g05", "rcv2_g08", "rcv2_g11", "rcv2_g14"}
    trajectory_repeat_groups = {"rcv2_g03", "rcv2_g06", "rcv2_g09", "rcv2_g12", "rcv2_g15"}

    for i in range(1, 16):
        group = f"rcv2_g{i:02d}"
        clean_id = f"{group}_s01"
        day24_plan.append({"plan_row_id": clean_id, "slot_role": "clean_control"})
        day24_records.append({
            "plan_row_id": clean_id, "pair_group_id": group,
            "selected_canonical": "true", "technical_valid": "true",
            "experimental_valid": "true", "task_success": "true",
            "intervention_applied": "false",
        })

        target_ids = [f"{group}_s02"]
        if group in target_repeat_groups:
            target_ids.append(f"{group}_s05")
        for pid in target_ids:
            day24_plan.append({"plan_row_id": pid, "slot_role": "controlled_cause"})
            day24_records.append({
                "plan_row_id": pid, "pair_group_id": group,
                "selected_canonical": "true", "technical_valid": "true",
                "experimental_valid": "true", "task_success": "false",
                "intervention_applied": "true",
            })

        gripper_ids = [f"{group}_s03"]
        if group in gripper_repeat_groups:
            gripper_ids.append(f"{group}_s05")
        for pid in gripper_ids:
            day25_plan.append({"plan_row_id": pid})
            day25_records.append({
                "plan_row_id": pid, "pair_group_id": group,
                "selected_canonical": "true", "technical_valid": "true",
                "experimental_valid": "true", "task_success": "false",
            })

        trajectory_ids = [f"{group}_s04"]
        if group in trajectory_repeat_groups:
            trajectory_ids.append(f"{group}_s05")
        for pid in trajectory_ids:
            day26_plan.append({"plan_row_id": pid})
            day26_records.append({
                "plan_row_id": pid, "pair_group_id": group,
                "selected_canonical": "true", "technical_valid": "true",
                "experimental_valid": "true", "task_success": "false",
            })

    return day24_plan, day24_records, day25_plan, day25_records, day26_plan, day26_records


def test_precollection_state_is_75_of_90_and_no_recollection():
    d24p, d24r, d25p, d25r, d26p, d26r = make_preexisting()
    pre = validate_preexisting_final_slots(
        day24_plan_rows=d24p, day24_records=d24r,
        day25_plan_rows=d25p, day25_records=d25r,
        day26_plan_rows=d26p, day26_records=d26r,
    )
    plan = expected_day27_rows_from_day22(day22_rows())
    analysis = analyze_collection(
        plan_rows=plan, records=[], preexisting=pre, raw_root=None
    )
    assert analysis["new_insufficient_candidate_canonical_count"] == 0
    assert analysis["clean_anchor_count"] == 15
    assert analysis["controlled_canonical_count"] == 60
    assert analysis["eligible_target_episode_count_through_day27"] == 75
    assert analysis["complete_pair_group_count"] == 0
    assert analysis["preexisting_recollection_required"] is False
    assert analysis["new_clean_collection_required"] is False
    assert analysis["status"] == "in_progress"


def test_final_state_reaches_90_and_allows_retained_recollection_attempts():
    d24p, d24r, d25p, d25r, d26p, d26r = make_preexisting()
    pre = validate_preexisting_final_slots(
        day24_plan_rows=d24p, day24_records=d24r,
        day25_plan_rows=d25p, day25_records=d25r,
        day26_plan_rows=d26p, day26_records=d26r,
    )
    plan = expected_day27_rows_from_day22(day22_rows())
    records = []
    for i, row in enumerate(plan, start=1):
        records.append({
            "plan_row_id": row["plan_row_id"],
            "pair_group_id": row["pair_group_id"],
            "selected_canonical": "true",
            "technical_valid": "true",
            "experimental_valid": "true",
            "task_success": "false",
        })
    # Retained successful noncanonical attempt for g01 is allowed and requires recollection.
    records.append({
        "plan_row_id": "rcv2_g01_s06",
        "pair_group_id": "rcv2_g01",
        "selected_canonical": "false",
        "technical_valid": "true",
        "experimental_valid": "false",
        "task_success": "true",
    })
    analysis = analyze_collection(
        plan_rows=plan, records=records, preexisting=pre, raw_root=None
    )
    assert analysis["new_insufficient_candidate_canonical_count"] == 15
    assert analysis["insufficient_candidate_failure_count"] == 15
    assert analysis["eligible_target_episode_count_through_day27"] == 90
    assert analysis["complete_pair_group_count"] == 15
    assert analysis["recollection_attempt_count"] == 1
    assert analysis["experimental_exclusion_attempt_count"] == 1
    assert validate_final_analysis(analysis) == []


def test_v2_variant_rotation_is_balanced_5_each():
    rows = expected_day27_rows_from_day22(day22_rows())
    variants = [r["ambiguity_protocol"].split(":", 1)[1] for r in rows]
    assert variants.count("target_mild_20mm_forward") == 5
    assert variants.count("gripper_late_30_40mm_upward_progress") == 5
    assert variants.count("trajectory_mild_25mm_forward") == 5


def test_day27_does_not_prejudge_answerability():
    d24p, d24r, d25p, d25r, d26p, d26r = make_preexisting()
    pre = validate_preexisting_final_slots(
        day24_plan_rows=d24p, day24_records=d24r,
        day25_plan_rows=d25p, day25_records=d25r,
        day26_plan_rows=d26p, day26_records=d26r,
    )
    plan = expected_day27_rows_from_day22(day22_rows())
    analysis = analyze_collection(plan_rows=plan, records=[], preexisting=pre, raw_root=None)
    assert analysis["ambiguity_protocol"]["answerability_prejudged_on_day27"] is False
