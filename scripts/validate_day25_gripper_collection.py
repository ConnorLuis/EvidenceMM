#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from evidencemm.day24_target_collection import assert_frozen_day22_plan,load_csv
from evidencemm.day25_gripper_collection import (
    analyze_collection,expected_day25_rows_from_day22,selected_day24_clean_anchors,
    validate_day25_plan_shape,validate_final_analysis)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--day22-plan",type=Path,default=Path("data/protocol/day22_root_cause_collection_plan.csv"))
    p.add_argument("--day25-plan",type=Path,default=Path("data/protocol/day25_gripper_collection_plan.csv"))
    p.add_argument("--records",type=Path,default=Path("data/protocol/day25_gripper_collection_records.csv"))
    p.add_argument("--day24-plan",type=Path,default=Path("data/protocol/day24_target_collection_plan.csv"))
    p.add_argument("--day24-records",type=Path,default=Path("data/protocol/day24_target_collection_records.csv"))
    p.add_argument("--raw-root",type=Path,default=Path(
        "/mnt/c/Users/Administrator/projects/embodied-agent-arm/outputs/episodes_root_cause_v2_final"))
    return p.parse_args()

def comparable(rows):
    fields=("plan_row_id","pair_group_id","day22_slot_index","slot_role",
            "planned_physical_cause","planned_intervention_type","repeat_slot",
            "expected_task_outcome","parameter_direction","parameter_min","parameter_max",
            "parameter_unit","operational_measurement","clean_anchor_plan_row_id")
    return [tuple(str(r.get(f,"")) for f in fields) for r in rows]

def main():
    a=args()
    assert_frozen_day22_plan(a.day22_plan)
    day22=load_csv(a.day22_plan)
    actual=load_csv(a.day25_plan)
    expected=expected_day25_rows_from_day22(day22)
    validate_day25_plan_shape(actual)
    if comparable(actual)!=comparable(expected):
        raise SystemExit("DAY25 VALIDATION FAIL: derived Day25 plan does not match frozen Day22 plan")
    anchors=selected_day24_clean_anchors(load_csv(a.day24_plan),load_csv(a.day24_records))
    analysis=analyze_collection(plan_rows=actual,records=load_csv(a.records),
                                clean_anchors=anchors,raw_root=a.raw_root)
    errors=validate_final_analysis(analysis)
    if errors:
        print("===== DAY25 GRIPPER COLLECTION: NOT READY =====")
        for e in errors: print("FAIL:",e)
        print(f"CURRENT: gripper={analysis['new_gripper_canonical_count']}/20 "
              f"failures={analysis['gripper_failure_count']}/20 "
              f"clean_anchors={analysis['clean_anchor_count']}/15 "
              f"groups={analysis['complete_pair_group_count']}/15")
        return 2
    print("===== DAY25 GRIPPER COLLECTION: PASS =====")
    print("frozen_day22_plan_sha256: PASS")
    print("derived_gripper_plan: 20/20 PASS")
    print("day24_clean_anchors: 15/15 PASS")
    print("pair_groups: 15/15 PASS")
    print("gripper_failure: 20/20 PASS")
    print("gripper_parameter: late close after 30-40 mm upward progress PASS")
    print("recorder: 20/20 new episode_recorder_v7 technical PASS")
    print("new_clean_collection_required: false PASS")
    print("future_split_materialized: false PASS")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
