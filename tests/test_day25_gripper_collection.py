import pytest

from evidencemm.day24_target_collection import TechnicalAudit
from evidencemm.day25_gripper_collection import (
    GRIPPER_REPEAT_GROUPS,
    analyze_collection,
    build_record,
    canonical_records,
    evaluate_experimental_validity,
    expected_day25_rows_from_day22,
    selected_day24_clean_anchors,
    validate_day25_plan_shape,
    validate_final_analysis,
)

def _day22_rows():
    rows=[]
    gripper=set(GRIPPER_REPEAT_GROUPS)
    target={"rcv2_g01","rcv2_g04","rcv2_g07","rcv2_g10","rcv2_g13"}
    for i in range(1,16):
        g=f"rcv2_g{i:02d}"
        rows += [
            {"plan_row_id":f"{g}_s01","pair_group_id":g,"slot_index":"1","slot_role":"clean_control","planned_physical_cause":"none_clean","planned_intervention_type":"none","repeat_slot":"False"},
            {"plan_row_id":f"{g}_s03","pair_group_id":g,"slot_index":"3","slot_role":"controlled_cause","planned_physical_cause":"gripper_close_timing","planned_intervention_type":"manual_gripper_close_timing_shift","repeat_slot":"False"},
        ]
        if g in target:
            cause,typ="target_offset_or_perception","object_target_pose_offset"
        elif g in gripper:
            cause,typ="gripper_close_timing","manual_gripper_close_timing_shift"
        else:
            cause,typ="trajectory_execution_deviation","manual_bounded_trajectory_deviation"
        rows.append({"plan_row_id":f"{g}_s05","pair_group_id":g,"slot_index":"5","slot_role":"controlled_cause","planned_physical_cause":cause,"planned_intervention_type":typ,"repeat_slot":"True"})
    return rows

def _audit(ep="ep",valid=True):
    return TechnicalAudit(
        episode_id=ep,raw_episode_relpath=ep,
        recorder_script_version="episode_recorder_v7",
        technical_valid=valid,recorder_overall_pass=valid,
        failed_checks=[] if valid else ["x"],sample_count=900,csv_row_count=900,
        front_image_count=900,wrist_image_count=900,duration_seconds=59.93,
        max_tracking_error=10.0)

def _anchor(g):
    return {"plan_row_id":f"{g}_s01","pair_group_id":g,
            "technical_valid":"true","experimental_valid":"true",
            "task_success":"true","intervention_applied":"false",
            "selected_canonical":"true","episode_id":f"{g}_clean",
            "raw_episode_relpath":f"{g}_clean"}

def test_derive_plan_20():
    rows=expected_day25_rows_from_day22(_day22_rows())
    assert len(rows)==20
    validate_day25_plan_shape(rows)

def test_repeat_groups_exact():
    rows=expected_day25_rows_from_day22(_day22_rows())
    assert {r["pair_group_id"] for r in rows if r["repeat_slot"]=="true"} == set(GRIPPER_REPEAT_GROUPS)

def test_valid_frozen_failure():
    row=expected_day25_rows_from_day22(_day22_rows())[0]
    ok,reason=evaluate_experimental_validity(
        plan_row=row,technical_valid=True,task_success=False,
        intervention_applied=True,single_primary_intervention=True,
        changed_factor_observable=True,phase_proxy_met=True,
        safety_abort=False,hardware_fault=False)
    assert ok and reason==""

@pytest.mark.parametrize("field,reason",[
    ("task_success","gripper_intervention_did_not_induce_failure"),
    ("intervention_applied","declared_gripper_intervention_not_applied"),
    ("single_primary_intervention","multiple_primary_interventions"),
    ("changed_factor_observable","changed_factor_not_observable"),
    ("phase_proxy_met","late_close_phase_proxy_not_met"),
])
def test_reject_protocol_drift(field,reason):
    row=expected_day25_rows_from_day22(_day22_rows())[0]
    kw=dict(plan_row=row,technical_valid=True,task_success=False,
            intervention_applied=True,single_primary_intervention=True,
            changed_factor_observable=True,phase_proxy_met=True,
            safety_abort=False,hardware_fault=False)
    kw[field]=not kw[field]
    ok,got=evaluate_experimental_validity(**kw)
    assert not ok and got==reason

def test_build_record_freezes_range():
    row=expected_day25_rows_from_day22(_day22_rows())[0]
    rec=build_record(plan_row=row,audit=_audit(),attempt_index=1,task_success=False,
        intervention_applied=True,single_primary_intervention=True,
        changed_factor_observable=True,phase_proxy_met=True,safety_abort=False,
        hardware_fault=False,selected_canonical=True,operator_notes="")
    assert rec["selected_canonical"]=="true"
    assert rec["parameter_direction"]=="late"
    assert rec["parameter_min"]=="30" and rec["parameter_max"]=="40"

def test_day24_anchor_selection():
    plan=[{"plan_row_id":f"rcv2_g{i:02d}_s01","slot_role":"clean_control"} for i in range(1,16)]
    anchors=[_anchor(f"rcv2_g{i:02d}") for i in range(1,16)]
    assert len(selected_day24_clean_anchors(plan,anchors))==15

def test_duplicate_canonical_rejected():
    plan=expected_day25_rows_from_day22(_day22_rows())
    rec={"plan_row_id":plan[0]["plan_row_id"],"selected_canonical":"true"}
    with pytest.raises(ValueError):
        canonical_records(plan,[rec.copy(),rec.copy()])

def test_complete_analysis_without_raw_root():
    plan=expected_day25_rows_from_day22(_day22_rows())
    records=[]
    for n,row in enumerate(plan,1):
        records.append(build_record(plan_row=row,audit=_audit(f"ep{n}"),attempt_index=1,
            task_success=False,intervention_applied=True,single_primary_intervention=True,
            changed_factor_observable=True,phase_proxy_met=True,safety_abort=False,
            hardware_fault=False,selected_canonical=True,operator_notes=""))
    anchors=[_anchor(f"rcv2_g{i:02d}") for i in range(1,16)]
    a=analyze_collection(plan_rows=plan,records=records,clean_anchors=anchors,raw_root=None)
    assert a["status"]=="complete"
    assert a["new_gripper_canonical_count"]==20
    assert a["gripper_failure_count"]==20
    assert a["clean_anchor_count"]==15
    assert a["complete_pair_group_count"]==15
    assert validate_final_analysis(a)==[]

def test_empty_collection_in_progress():
    plan=expected_day25_rows_from_day22(_day22_rows())
    anchors=[_anchor(f"rcv2_g{i:02d}") for i in range(1,16)]
    a=analyze_collection(plan_rows=plan,records=[],clean_anchors=anchors,raw_root=None)
    assert a["status"]=="in_progress"
    assert a["new_gripper_canonical_count"]==0
    assert a["complete_pair_group_count"]==0
