#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evidencemm.day24_target_collection import load_csv
from evidencemm.day27_insufficient_evidence_collection import (
    analyze_collection,
    validate_preexisting_final_slots,
)


def args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "data/protocol/day27_insufficient_evidence_collection_plan.csv"
        ),
    )
    p.add_argument(
        "--records",
        type=Path,
        default=Path(
            "data/protocol/day27_insufficient_evidence_collection_records.csv"
        ),
    )
    p.add_argument(
        "--day24-plan",
        type=Path,
        default=Path("data/protocol/day24_target_collection_plan.csv"),
    )
    p.add_argument(
        "--day24-records",
        type=Path,
        default=Path("data/protocol/day24_target_collection_records.csv"),
    )
    p.add_argument(
        "--day25-plan",
        type=Path,
        default=Path("data/protocol/day25_gripper_collection_plan.csv"),
    )
    p.add_argument(
        "--day25-records",
        type=Path,
        default=Path("data/protocol/day25_gripper_collection_records.csv"),
    )
    p.add_argument(
        "--day26-plan",
        type=Path,
        default=Path("data/protocol/day26_trajectory_collection_plan.csv"),
    )
    p.add_argument(
        "--day26-records",
        type=Path,
        default=Path("data/protocol/day26_trajectory_collection_records.csv"),
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/mnt/c/Users/Administrator/projects/embodied-agent-arm/"
            "outputs/episodes_root_cause_v2_final"
        ),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/protocol/day27_insufficient_evidence_collection_analysis.json"
        ),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/protocol/day27_insufficient_evidence_collection_analysis.csv"
        ),
    )
    return p.parse_args()


def main():
    a = args()
    preexisting = validate_preexisting_final_slots(
        day24_plan_rows=load_csv(a.day24_plan),
        day24_records=load_csv(a.day24_records),
        day25_plan_rows=load_csv(a.day25_plan),
        day25_records=load_csv(a.day25_records),
        day26_plan_rows=load_csv(a.day26_plan),
        day26_records=load_csv(a.day26_records),
    )
    analysis = analyze_collection(
        plan_rows=load_csv(a.plan),
        records=load_csv(a.records),
        preexisting=preexisting,
        raw_root=a.raw_root,
    )
    a.output_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = (
        "plan_row_id",
        "pair_group_id",
        "episode_id",
        "task_success",
        "technical_valid",
        "duration_seconds",
        "max_tracking_error",
        "recorder_script_version",
    )
    with a.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(analysis["canonical_details"])

    print("===== DAY27 INSUFFICIENT-EVIDENCE COLLECTION ANALYSIS =====")
    print(
        f"insufficient_canonical="
        f"{analysis['new_insufficient_candidate_canonical_count']}/15 "
        f"failures={analysis['insufficient_candidate_failure_count']}/15 "
        f"clean_anchors={analysis['clean_anchor_count']}/15 "
        f"controlled={analysis['controlled_canonical_count']}/60 "
        f"complete_groups={analysis['complete_pair_group_count']}/15"
    )
    print(
        f"new_attempts={analysis['new_attempt_count']} "
        f"recollection_attempts={analysis['recollection_attempt_count']} "
        f"technical_exclusions={analysis['technical_exclusion_attempt_count']} "
        f"experimental_exclusions={analysis['experimental_exclusion_attempt_count']}"
    )
    print(
        "preexisting_recollection_required=",
        analysis["preexisting_recollection_required"],
    )
    print("new_clean_collection_required=", analysis["new_clean_collection_required"])
    print(
        "eligible_target_episode_count_through_day27=",
        analysis["eligible_target_episode_count_through_day27"],
    )
    print("status=", analysis["status"])
    print("JSON:", a.output_json)
    print("CSV :", a.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
