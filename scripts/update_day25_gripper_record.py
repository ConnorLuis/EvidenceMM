#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from evidencemm.day24_target_collection import (
    audit_episode, load_csv, next_attempt_index, parse_bool,
    selected_for_plan, write_csv,
)
from evidencemm.day25_gripper_collection import RECORD_FIELDS, build_record

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--plan-row", required=True)
    p.add_argument("--episode-id", required=True)
    p.add_argument("--task-success", required=True, choices=("true","false"))
    p.add_argument("--applied", required=True, choices=("true","false"))
    p.add_argument("--single-primary", default="true", choices=("true","false"))
    p.add_argument("--observable", default="true", choices=("true","false"))
    p.add_argument("--phase-proxy-met", default="true", choices=("true","false"))
    p.add_argument("--safety-abort", default="false", choices=("true","false"))
    p.add_argument("--hardware-fault", default="false", choices=("true","false"))
    p.add_argument("--notes", default="")
    p.add_argument("--raw-root", type=Path, default=Path(
        "/mnt/c/Users/Administrator/projects/embodied-agent-arm/"
        "outputs/episodes_root_cause_v2_final"))
    p.add_argument("--plan", type=Path, default=Path(
        "data/protocol/day25_gripper_collection_plan.csv"))
    p.add_argument("--records", type=Path, default=Path(
        "data/protocol/day25_gripper_collection_records.csv"))
    return p.parse_args()

def main():
    a=args()
    plan=load_csv(a.plan)
    matches=[r for r in plan if r["plan_row_id"]==a.plan_row]
    if len(matches)!=1:
        raise SystemExit(f"unknown or duplicate plan row: {a.plan_row}")
    plan_row=matches[0]
    existing=load_csv(a.records)
    if selected_for_plan(existing,a.plan_row):
        raise SystemExit(f"{a.plan_row} already has a selected canonical episode")
    audit=audit_episode(a.raw_root/a.episode_id)
    record=build_record(
        plan_row=plan_row, audit=audit,
        attempt_index=next_attempt_index(existing,a.plan_row),
        task_success=parse_bool(a.task_success),
        intervention_applied=parse_bool(a.applied),
        single_primary_intervention=parse_bool(a.single_primary),
        changed_factor_observable=parse_bool(a.observable),
        phase_proxy_met=parse_bool(a.phase_proxy_met),
        safety_abort=parse_bool(a.safety_abort),
        hardware_fault=parse_bool(a.hardware_fault),
        selected_canonical=True, operator_notes=a.notes)
    existing.append(record)
    write_csv(a.records,existing,RECORD_FIELDS)
    print("===== DAY25 RECORD UPDATED =====")
    for key in ("plan_row_id","episode_id","attempt_index","technical_valid",
                "experimental_valid","phase_proxy_met","selected_canonical"):
        print(f"{key:20s}=",record[key])
    print("exclusion_reason     =",record["exclusion_reason"] or "<none>")
    if record["selected_canonical"]!="true":
        print("RESULT: registered as non-canonical attempt. Recollect SAME plan row with SAME frozen late-close phase proxy.")
        return 2
    print("RESULT: canonical slot accepted.")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
