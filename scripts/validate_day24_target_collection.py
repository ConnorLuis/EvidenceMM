#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from evidencemm.day24_target_collection import (
    analyze_collection,
    assert_frozen_day22_plan,
    expected_day24_rows_from_day22,
    load_csv,
    validate_day24_plan_shape,
    validate_final_analysis,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--day22-plan",
        type=Path,
        default=Path("data/protocol/day22_root_cause_collection_plan.csv"),
    )
    p.add_argument(
        "--day24-plan",
        type=Path,
        default=Path("data/protocol/day24_target_collection_plan.csv"),
    )
    p.add_argument(
        "--records",
        type=Path,
        default=Path("data/protocol/day24_target_collection_records.csv"),
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/mnt/c/Users/Administrator/projects/embodied-agent-arm/"
            "outputs/episodes_root_cause_v2_final"
        ),
    )
    return p.parse_args()


def comparable(rows):
    fields = (
        "plan_row_id","pair_group_id","day22_slot_index","slot_role",
        "planned_physical_cause","planned_intervention_type","repeat_slot",
        "expected_task_outcome","parameter_direction","parameter_value",
        "parameter_unit",
    )
    return [
        tuple(str(row.get(field, "")) for field in fields)
        for row in rows
    ]


def main() -> int:
    a = args()

    assert_frozen_day22_plan(a.day22_plan)
    day22 = load_csv(a.day22_plan)
    actual_plan = load_csv(a.day24_plan)
    expected_plan = expected_day24_rows_from_day22(day22)

    validate_day24_plan_shape(actual_plan)
    if comparable(actual_plan) != comparable(expected_plan):
        raise SystemExit(
            "DAY24 VALIDATION FAIL: derived Day24 plan does not match frozen Day22 plan"
        )

    records = load_csv(a.records)
    analysis = analyze_collection(
        plan_rows=actual_plan,
        records=records,
        raw_root=a.raw_root,
    )
    errors = validate_final_analysis(analysis)

    if errors:
        print("===== DAY24 TARGET COLLECTION: NOT READY =====")
        for error in errors:
            print("FAIL:", error)
        print(
            f"CURRENT: canonical={analysis['canonical_episode_count']}/35 "
            f"clean_success={analysis['clean_success_count']}/15 "
            f"target_failure={analysis['target_failure_count']}/20 "
            f"groups={analysis['complete_pair_group_count']}/15"
        )
        return 2

    print("===== DAY24 TARGET COLLECTION: PASS =====")
    print("frozen_day22_plan_sha256: PASS")
    print("derived_plan: 35/35 PASS")
    print("pair_groups: 15/15 PASS")
    print("clean_success: 15/15 PASS")
    print("target_failure: 20/20 PASS")
    print("target_parameter: follower_forward 40 mm PASS")
    print("recorder: 35/35 episode_recorder_v7 technical PASS")
    print("future_split_materialized: false PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
