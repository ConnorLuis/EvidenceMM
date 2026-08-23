\
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evidencemm.root_cause_pilot import RECORD_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update one Day23 pilot record row without editing CSV by hand."
    )
    parser.add_argument("--records", type=Path, default=Path("data/protocol/day23_pilot_records.csv"))
    parser.add_argument("--row", required=True, dest="pilot_row_id")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--raw-relpath", required=True)
    parser.add_argument("--recorder-pass", required=True, choices=["true", "false"])
    parser.add_argument("--failed-checks", default="")
    parser.add_argument("--task-success", required=True, choices=["true", "false"])
    parser.add_argument("--predeclared", required=True, choices=["true", "false"])
    parser.add_argument("--applied", required=True, choices=["true", "false"])
    parser.add_argument("--single-primary", default="true", choices=["true", "false"])
    parser.add_argument("--direction", default="")
    parser.add_argument("--value", default="")
    parser.add_argument("--unit", default="")
    parser.add_argument("--observable", required=True, choices=["true", "false"])
    parser.add_argument("--modalities", default="")
    parser.add_argument("--gripper-close-verified", default="")
    parser.add_argument("--safety-abort", default="false", choices=["true", "false"])
    parser.add_argument("--hardware-fault", default="false", choices=["true", "false"])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    path = args.records
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RECORD_COLUMNS:
            raise ValueError("record CSV columns differ from Day23 template")
        rows = list(reader)

    matches = [row for row in rows if row["pilot_row_id"] == args.pilot_row_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row {args.pilot_row_id}")
    row = matches[0]
    row.update(
        {
            "episode_id": args.episode_id,
            "raw_episode_relpath": args.raw_relpath,
            "recorder_overall_pass": args.recorder_pass,
            "failed_checks": args.failed_checks,
            "task_success": args.task_success,
            "intervention_predeclared": args.predeclared,
            "intervention_applied": args.applied,
            "single_primary_intervention": args.single_primary,
            "parameter_direction": args.direction,
            "parameter_value": args.value,
            "parameter_unit": args.unit,
            "changed_factor_observable": args.observable,
            "observable_modalities": args.modalities,
            "gripper_transition_verified_as_grasp_close": args.gripper_close_verified,
            "safety_abort": args.safety_abort,
            "hardware_fault": args.hardware_fault,
            "operator_notes": args.notes,
        }
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RECORD_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"UPDATED {args.pilot_row_id} -> {args.episode_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
