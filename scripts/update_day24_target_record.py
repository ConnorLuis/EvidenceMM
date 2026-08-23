#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evidencemm.day24_target_collection import (
    RECORD_FIELDS,
    audit_episode,
    build_record,
    load_csv,
    next_attempt_index,
    parse_bool,
    selected_for_plan,
    write_csv,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan-row", required=True)
    p.add_argument("--episode-id", required=True)
    p.add_argument("--task-success", required=True, choices=("true", "false"))
    p.add_argument("--applied", required=True, choices=("true", "false"))
    p.add_argument("--single-primary", default="true", choices=("true", "false"))
    p.add_argument("--observable", default="true", choices=("true", "false"))
    p.add_argument("--safety-abort", default="false", choices=("true", "false"))
    p.add_argument("--hardware-fault", default="false", choices=("true", "false"))
    p.add_argument("--notes", default="")
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/mnt/c/Users/Administrator/projects/embodied-agent-arm/"
            "outputs/episodes_root_cause_v2_final"
        ),
    )
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
    return p.parse_args()


def main() -> int:
    a = args()
    plan = load_csv(a.plan)
    matches = [row for row in plan if row["plan_row_id"] == a.plan_row]
    if len(matches) != 1:
        raise SystemExit(f"unknown or duplicate plan row: {a.plan_row}")
    plan_row = matches[0]

    existing = load_csv(a.records)
    if selected_for_plan(existing, a.plan_row):
        raise SystemExit(
            f"{a.plan_row} already has a selected canonical episode; "
            "do not overwrite frozen raw history"
        )

    episode_dir = a.raw_root / a.episode_id
    audit = audit_episode(episode_dir)
    attempt = next_attempt_index(existing, a.plan_row)

    record = build_record(
        plan_row=plan_row,
        audit=audit,
        attempt_index=attempt,
        task_success=parse_bool(a.task_success),
        intervention_applied=parse_bool(a.applied),
        single_primary_intervention=parse_bool(a.single_primary),
        changed_factor_observable=parse_bool(a.observable),
        safety_abort=parse_bool(a.safety_abort),
        hardware_fault=parse_bool(a.hardware_fault),
        selected_canonical=True,
        operator_notes=a.notes,
    )

    existing.append(record)
    write_csv(a.records, existing, RECORD_FIELDS)

    print("===== DAY24 RECORD UPDATED =====")
    print("plan_row_id       =", record["plan_row_id"])
    print("episode_id        =", record["episode_id"])
    print("attempt_index     =", record["attempt_index"])
    print("technical_valid   =", record["technical_valid"])
    print("experimental_valid=", record["experimental_valid"])
    print("selected_canonical=", record["selected_canonical"])
    print("exclusion_reason  =", record["exclusion_reason"] or "<none>")

    if record["selected_canonical"] != "true":
        print(
            "RESULT: retained as non-canonical attempt. "
            "Recollect the SAME plan row using the SAME frozen parameter."
        )
        return 2

    print("RESULT: canonical slot accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
