#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from evidencemm.day24_target_collection import (
    audit_episode,
    load_csv,
    next_attempt_index,
    parse_bool,
    selected_for_plan,
)
from evidencemm.day27_insufficient_evidence_collection import (
    RECORD_FIELDS,
    build_record,
    write_csv_lf,
)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan-row", required=True)
    p.add_argument("--episode-id", required=True)
    p.add_argument("--task-success", required=True, choices=("true", "false"))
    p.add_argument("--variant", required=True)
    p.add_argument("--protocol-followed", default="true", choices=("true", "false"))
    p.add_argument(
        "--known-cause-intervention",
        default="false",
        choices=("true", "false"),
    )
    p.add_argument(
        "--intentional-failure-injection",
        default="false",
        choices=("true", "false"),
    )
    p.add_argument(
        "--multiple-primary-interventions",
        default="false",
        choices=("true", "false"),
    )
    p.add_argument("--scene-comparable", default="true", choices=("true", "false"))
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
    return p.parse_args()


def main():
    a = args()
    plan = load_csv(a.plan)
    matches = [row for row in plan if row["plan_row_id"] == a.plan_row]
    if len(matches) != 1:
        raise SystemExit(f"unknown or duplicate Day27 plan row: {a.plan_row}")
    if selected_for_plan(load_csv(a.records), a.plan_row):
        raise SystemExit(f"{a.plan_row} already has a selected canonical episode")
    expected_variant = matches[0]["ambiguity_protocol"].split(":", 1)[1]
    if a.variant != expected_variant:
        raise SystemExit(
            f"variant mismatch for {a.plan_row}: expected {expected_variant}, got {a.variant}"
        )

    existing = load_csv(a.records)
    audit = audit_episode(a.raw_root / a.episode_id)
    record = build_record(
        plan_row=matches[0],
        audit=audit,
        attempt_index=next_attempt_index(existing, a.plan_row),
        task_success=parse_bool(a.task_success),
        ambiguity_protocol_followed=parse_bool(a.protocol_followed),
        deliberate_known_cause_intervention=parse_bool(a.known_cause_intervention),
        intentional_failure_injection=parse_bool(a.intentional_failure_injection),
        multiple_primary_interventions=parse_bool(
            a.multiple_primary_interventions
        ),
        scene_comparable=parse_bool(a.scene_comparable),
        safety_abort=parse_bool(a.safety_abort),
        hardware_fault=parse_bool(a.hardware_fault),
        selected_canonical=True,
        operator_notes=a.notes,
    )
    existing.append(record)
    write_csv_lf(a.records, existing, RECORD_FIELDS)

    print("===== DAY27 RECORD UPDATED =====")
    for key in (
        "plan_row_id",
        "episode_id",
        "attempt_index",
        "technical_valid",
        "experimental_valid",
        "selected_canonical",
    ):
        print(f"{key:26s}=", record[key])
    print("exclusion_reason           =", record["exclusion_reason"] or "<none>")
    if record["selected_canonical"] != "true":
        print(
            "RESULT: raw attempt retained as non-canonical. "
            "Recollect SAME s06 with the SAME frozen Day27-v2 assigned variant."
        )
        return 2
    print("RESULT: canonical s06 candidate accepted; final answerability is NOT assigned until Day29 blind human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
