#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evidencemm.day24_target_collection import (
    analyze_collection,
    load_csv,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--plan",
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
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/protocol/day24_target_collection_analysis.json"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/protocol/day24_target_collection_analysis.csv"),
    )
    return p.parse_args()


def main() -> int:
    a = args()
    plan = load_csv(a.plan)
    records = load_csv(a.records)
    analysis = analyze_collection(
        plan_rows=plan,
        records=records,
        raw_root=a.raw_root,
    )
    a.output_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    details = analysis["canonical_details"]
    fields = (
        "plan_row_id","pair_group_id","episode_id","task_success",
        "technical_valid","duration_seconds","max_tracking_error",
        "recorder_script_version",
    )
    with a.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)

    print("===== DAY24 TARGET COLLECTION ANALYSIS =====")
    print(
        f"canonical={analysis['canonical_episode_count']}/35 "
        f"clean_success={analysis['clean_success_count']}/15 "
        f"target_failure={analysis['target_failure_count']}/20 "
        f"complete_groups={analysis['complete_pair_group_count']}/15"
    )
    print(
        f"attempts={analysis['attempt_count']} "
        f"technical_exclusions={analysis['technical_exclusion_attempt_count']} "
        f"experimental_exclusions={analysis['experimental_exclusion_attempt_count']}"
    )
    print("status=", analysis["status"])
    print("JSON:", a.output_json)
    print("CSV :", a.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
