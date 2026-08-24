from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from evidencemm.day24_target_collection import (
    EXPECTED_PAIR_GROUPS,
    ACQUISITION_CONFIGURATION,
    CAMERA_SETUP,
    NOMINAL_START_CONDITION,
    OBJECT_IDENTITY,
    SCENE_SETUP_ID,
    TechnicalAudit,
    audit_episode,
    bool_text,
    parse_bool,
)

TRAJECTORY_REPEAT_GROUPS = ("rcv2_g03", "rcv2_g06", "rcv2_g09", "rcv2_g12", "rcv2_g15")
TRAJECTORY_DIRECTION = "follower_forward"
TRAJECTORY_MIN_MM = 40.0
TRAJECTORY_MAX_MM = 60.0
TRAJECTORY_UNIT = "mm"
TRAJECTORY_OPERATIONAL_MEASUREMENT = "marked_lateral_waypoint_offset"

PLAN_FIELDS = (
    "schema_version", "day26_sequence", "plan_row_id", "pair_group_id",
    "day22_slot_index", "slot_role", "planned_physical_cause",
    "planned_intervention_type", "repeat_slot", "expected_task_outcome",
    "parameter_direction", "parameter_min", "parameter_max", "parameter_unit",
    "operational_measurement", "clean_anchor_plan_row_id", "collection_status",
)

RECORD_FIELDS = (
    "schema_version", "plan_row_id", "pair_group_id", "attempt_index",
    "episode_id", "raw_episode_relpath", "recorder_script_version",
    "technical_valid", "recorder_overall_pass", "failed_checks",
    "task_success", "intervention_applied", "single_primary_intervention",
    "changed_factor_observable", "deviation_proxy_met", "parameter_direction",
    "parameter_min", "parameter_max", "parameter_unit",
    "operational_measurement", "scene_setup_id", "object_identity",
    "nominal_start_condition", "camera_setup", "acquisition_configuration",
    "safety_abort", "hardware_fault", "experimental_valid",
    "selected_canonical", "exclusion_reason", "operator_notes",
)


def write_csv_lf(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def expected_day26_rows_from_day22(day22_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    sequence = 1
    for pair_index in range(1, 16):
        group = f"rcv2_g{pair_index:02d}"
        members = [row for row in day22_rows if row["pair_group_id"] == group]
        primary = [
            row for row in members
            if row["slot_index"] == "4"
            and row["planned_physical_cause"] == "trajectory_execution_deviation"
        ]
        repeat = [
            row for row in members
            if row["slot_index"] == "5"
            and row["planned_physical_cause"] == "trajectory_execution_deviation"
        ]
        if len(primary) != 1:
            raise ValueError(f"Day22 group {group} missing trajectory-primary slot")
        for src in primary + repeat:
            selected.append({
                "schema_version": "evidencemm_day26_trajectory_collection_plan_v1",
                "day26_sequence": str(sequence),
                "plan_row_id": src["plan_row_id"],
                "pair_group_id": group,
                "day22_slot_index": src["slot_index"],
                "slot_role": src["slot_role"],
                "planned_physical_cause": src["planned_physical_cause"],
                "planned_intervention_type": src["planned_intervention_type"],
                "repeat_slot": str(src["repeat_slot"]).lower(),
                "expected_task_outcome": "failure",
                "parameter_direction": TRAJECTORY_DIRECTION,
                "parameter_min": f"{TRAJECTORY_MIN_MM:g}",
                "parameter_max": f"{TRAJECTORY_MAX_MM:g}",
                "parameter_unit": TRAJECTORY_UNIT,
                "operational_measurement": TRAJECTORY_OPERATIONAL_MEASUREMENT,
                "clean_anchor_plan_row_id": f"{group}_s01",
                "collection_status": "pending",
            })
            sequence += 1
    return selected


def validate_day26_plan_shape(rows: list[dict[str, str]]) -> None:
    if len(rows) != 20:
        raise ValueError(f"Day26 plan must contain 20 rows, got {len(rows)}")
    if {row["pair_group_id"] for row in rows} != set(EXPECTED_PAIR_GROUPS):
        raise ValueError("Day26 plan pair-group set mismatch")

    repeat = [row for row in rows if parse_bool(row["repeat_slot"])]
    if len(repeat) != 5:
        raise ValueError(f"expected 5 trajectory repeats, got {len(repeat)}")
    if {row["pair_group_id"] for row in repeat} != set(TRAJECTORY_REPEAT_GROUPS):
        raise ValueError("trajectory repeat groups mismatch")

    for group in EXPECTED_PAIR_GROUPS:
        group_rows = [row for row in rows if row["pair_group_id"] == group]
        expected = 2 if group in TRAJECTORY_REPEAT_GROUPS else 1
        if len(group_rows) != expected:
            raise ValueError(f"{group}: expected {expected} trajectory rows")

    for row in rows:
        if row["planned_physical_cause"] != "trajectory_execution_deviation":
            raise ValueError(f"{row['plan_row_id']}: unexpected cause")
        if row["planned_intervention_type"] != "manual_bounded_trajectory_deviation":
            raise ValueError(f"{row['plan_row_id']}: intervention type drift")
        if row["parameter_direction"] != TRAJECTORY_DIRECTION:
            raise ValueError(f"{row['plan_row_id']}: direction drift")
        if float(row["parameter_min"]) != TRAJECTORY_MIN_MM:
            raise ValueError(f"{row['plan_row_id']}: parameter_min drift")
        if float(row["parameter_max"]) != TRAJECTORY_MAX_MM:
            raise ValueError(f"{row['plan_row_id']}: parameter_max drift")
        if row["parameter_unit"] != TRAJECTORY_UNIT:
            raise ValueError(f"{row['plan_row_id']}: unit drift")
        if row["operational_measurement"] != TRAJECTORY_OPERATIONAL_MEASUREMENT:
            raise ValueError(f"{row['plan_row_id']}: measurement drift")
        if row["clean_anchor_plan_row_id"] != f"{row['pair_group_id']}_s01":
            raise ValueError(f"{row['plan_row_id']}: clean anchor drift")


def selected_day24_clean_anchors(
    day24_plan_rows: list[dict[str, str]],
    day24_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    clean_plan_ids = {
        row["plan_row_id"] for row in day24_plan_rows
        if row["slot_role"] == "clean_control"
    }
    selected = [
        row for row in day24_records
        if row["plan_row_id"] in clean_plan_ids
        and parse_bool(row.get("selected_canonical"))
    ]
    by_group: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_group.setdefault(row["pair_group_id"], []).append(row)
    if set(by_group) != set(EXPECTED_PAIR_GROUPS):
        raise ValueError("Day24 clean anchor pair-group set mismatch")
    for group, rows in by_group.items():
        if len(rows) != 1:
            raise ValueError(f"{group}: expected exactly one selected clean anchor")
        row = rows[0]
        if row["plan_row_id"] != f"{group}_s01":
            raise ValueError(f"{group}: selected clean anchor is not s01")
        if not parse_bool(row.get("technical_valid")):
            raise ValueError(f"{group}: clean anchor technical invalid")
        if not parse_bool(row.get("experimental_valid")):
            raise ValueError(f"{group}: clean anchor experimental invalid")
        if not parse_bool(row.get("task_success")):
            raise ValueError(f"{group}: clean anchor must be task success")
        if parse_bool(row.get("intervention_applied")):
            raise ValueError(f"{group}: clean anchor must have no intervention")
    return selected


def evaluate_experimental_validity(
    *, plan_row: dict[str, str], technical_valid: bool, task_success: bool,
    intervention_applied: bool, single_primary_intervention: bool,
    changed_factor_observable: bool, deviation_proxy_met: bool,
    safety_abort: bool, hardware_fault: bool,
) -> tuple[bool, str]:
    if not technical_valid:
        return False, "technical_exclusion"
    if safety_abort:
        return False, "safety_abort"
    if hardware_fault:
        return False, "hardware_fault"
    if plan_row["planned_physical_cause"] != "trajectory_execution_deviation":
        return False, "unexpected_day26_cause"
    if not intervention_applied:
        return False, "declared_trajectory_intervention_not_applied"
    if not single_primary_intervention:
        return False, "multiple_primary_interventions"
    if not changed_factor_observable:
        return False, "changed_factor_not_observable"
    if not deviation_proxy_met:
        return False, "trajectory_deviation_proxy_not_met"
    if task_success:
        return False, "trajectory_intervention_did_not_induce_failure"
    return True, ""


def build_record(
    *, plan_row: dict[str, str], audit: TechnicalAudit, attempt_index: int,
    task_success: bool, intervention_applied: bool,
    single_primary_intervention: bool, changed_factor_observable: bool,
    deviation_proxy_met: bool, safety_abort: bool, hardware_fault: bool,
    selected_canonical: bool, operator_notes: str,
) -> dict[str, str]:
    experimental_valid, exclusion_reason = evaluate_experimental_validity(
        plan_row=plan_row, technical_valid=audit.technical_valid,
        task_success=task_success, intervention_applied=intervention_applied,
        single_primary_intervention=single_primary_intervention,
        changed_factor_observable=changed_factor_observable,
        deviation_proxy_met=deviation_proxy_met, safety_abort=safety_abort,
        hardware_fault=hardware_fault,
    )
    return {
        "schema_version": "evidencemm_day26_trajectory_collection_record_v1",
        "plan_row_id": plan_row["plan_row_id"],
        "pair_group_id": plan_row["pair_group_id"],
        "attempt_index": str(attempt_index),
        "episode_id": audit.episode_id,
        "raw_episode_relpath": audit.raw_episode_relpath,
        "recorder_script_version": audit.recorder_script_version,
        "technical_valid": bool_text(audit.technical_valid),
        "recorder_overall_pass": bool_text(audit.recorder_overall_pass),
        "failed_checks": ";".join(audit.failed_checks),
        "task_success": bool_text(task_success),
        "intervention_applied": bool_text(intervention_applied),
        "single_primary_intervention": bool_text(single_primary_intervention),
        "changed_factor_observable": bool_text(changed_factor_observable),
        "deviation_proxy_met": bool_text(deviation_proxy_met),
        "parameter_direction": TRAJECTORY_DIRECTION,
        "parameter_min": f"{TRAJECTORY_MIN_MM:g}",
        "parameter_max": f"{TRAJECTORY_MAX_MM:g}",
        "parameter_unit": TRAJECTORY_UNIT,
        "operational_measurement": TRAJECTORY_OPERATIONAL_MEASUREMENT,
        "scene_setup_id": SCENE_SETUP_ID,
        "object_identity": OBJECT_IDENTITY,
        "nominal_start_condition": NOMINAL_START_CONDITION,
        "camera_setup": CAMERA_SETUP,
        "acquisition_configuration": ACQUISITION_CONFIGURATION,
        "safety_abort": bool_text(safety_abort),
        "hardware_fault": bool_text(hardware_fault),
        "experimental_valid": bool_text(experimental_valid),
        "selected_canonical": bool_text(selected_canonical and experimental_valid),
        "exclusion_reason": exclusion_reason,
        "operator_notes": operator_notes,
    }


def canonical_records(plan_rows, records):
    selected = [row for row in records if parse_bool(row.get("selected_canonical"))]
    plan_ids = {row["plan_row_id"] for row in plan_rows}
    by_plan = {}
    for row in selected:
        if row["plan_row_id"] not in plan_ids:
            raise ValueError("records contain unknown plan_row_id")
        by_plan.setdefault(row["plan_row_id"], []).append(row)
    duplicates = [k for k, v in by_plan.items() if len(v) > 1]
    if duplicates:
        raise ValueError("more than one canonical record for: " + ", ".join(sorted(duplicates)))
    return selected


def analyze_collection(
    *, plan_rows: list[dict[str, str]], records: list[dict[str, str]],
    clean_anchors: list[dict[str, str]], raw_root: Path | None = None,
) -> dict[str, Any]:
    validate_day26_plan_shape(plan_rows)
    canonical = canonical_records(plan_rows, records)
    canonical_by_id = {r["plan_row_id"]: r for r in canonical}
    anchor_by_group = {r["pair_group_id"]: r for r in clean_anchors}

    per_group = {}
    for group in EXPECTED_PAIR_GROUPS:
        expected_rows = [r for r in plan_rows if r["pair_group_id"] == group]
        selected_rows = [r for r in expected_rows if r["plan_row_id"] in canonical_by_id]
        per_group[group] = {
            "clean_anchor_present": group in anchor_by_group,
            "expected_trajectory_slots": len(expected_rows),
            "canonical_trajectory_slots": len(selected_rows),
            "complete": group in anchor_by_group and len(selected_rows) == len(expected_rows),
        }

    technical_exclusions = sum(1 for r in records if not parse_bool(r.get("technical_valid")))
    experimental_exclusions = sum(
        1 for r in records
        if parse_bool(r.get("technical_valid"))
        and not parse_bool(r.get("experimental_valid"))
    )

    canonical_details = []
    clean_anchor_details = []
    if raw_root is not None:
        for row in canonical:
            audit = audit_episode(raw_root / row["raw_episode_relpath"])
            canonical_details.append({
                "plan_row_id": row["plan_row_id"],
                "pair_group_id": row["pair_group_id"],
                "episode_id": row["episode_id"],
                "task_success": parse_bool(row["task_success"]),
                "technical_valid": audit.technical_valid,
                "duration_seconds": None if audit.duration_seconds is None else round(audit.duration_seconds, 6),
                "max_tracking_error": None if audit.max_tracking_error is None else round(audit.max_tracking_error, 4),
                "recorder_script_version": audit.recorder_script_version,
            })
        for row in clean_anchors:
            audit = audit_episode(raw_root / row["raw_episode_relpath"])
            clean_anchor_details.append({
                "plan_row_id": row["plan_row_id"],
                "pair_group_id": row["pair_group_id"],
                "episode_id": row["episode_id"],
                "technical_valid": audit.technical_valid,
                "recorder_script_version": audit.recorder_script_version,
            })

    return {
        "schema_version": "evidencemm_day26_trajectory_collection_analysis_v1",
        "status": "complete" if len(canonical) == 20 else "in_progress",
        "new_attempt_count": len(records),
        "new_trajectory_canonical_count": len(canonical),
        "expected_new_trajectory_canonical_count": 20,
        "trajectory_failure_count": sum(not parse_bool(r["task_success"]) for r in canonical),
        "clean_anchor_count": len(clean_anchors),
        "clean_anchor_success_count": sum(parse_bool(r["task_success"]) for r in clean_anchors),
        "technical_exclusion_attempt_count": technical_exclusions,
        "experimental_exclusion_attempt_count": experimental_exclusions,
        "pair_group_count": 15,
        "complete_pair_group_count": sum(x["complete"] for x in per_group.values()),
        "trajectory_parameter": {
            "direction": TRAJECTORY_DIRECTION,
            "operational_measurement": TRAJECTORY_OPERATIONAL_MEASUREMENT,
            "magnitude_range": [TRAJECTORY_MIN_MM, TRAJECTORY_MAX_MM],
            "unit": TRAJECTORY_UNIT,
            "measurement_precision": "operator_estimated_range",
        },
        "new_clean_collection_required": False,
        "future_split_materialized": False,
        "raw_attempt_retention_completeness_asserted": False,
        "per_group": per_group,
        "canonical_details": canonical_details,
        "clean_anchor_details": clean_anchor_details,
    }


def validate_final_analysis(analysis: dict[str, Any]) -> list[str]:
    errors = []
    expected = {
        "new_trajectory_canonical_count": 20,
        "trajectory_failure_count": 20,
        "clean_anchor_count": 15,
        "clean_anchor_success_count": 15,
        "complete_pair_group_count": 15,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            errors.append(f"{key}: expected {value}, got {analysis.get(key)}")
    if analysis.get("new_clean_collection_required") is not False:
        errors.append("Day26 must reuse Day24 clean anchors")
    if analysis.get("future_split_materialized") is not False:
        errors.append("future split must not be materialized on Day26")

    p = analysis.get("trajectory_parameter") or {}
    if p.get("direction") != TRAJECTORY_DIRECTION:
        errors.append("trajectory direction mismatch")
    if p.get("operational_measurement") != TRAJECTORY_OPERATIONAL_MEASUREMENT:
        errors.append("trajectory measurement mismatch")
    if list(p.get("magnitude_range") or []) != [TRAJECTORY_MIN_MM, TRAJECTORY_MAX_MM]:
        errors.append("trajectory magnitude range mismatch")
    if p.get("unit") != TRAJECTORY_UNIT:
        errors.append("trajectory unit mismatch")

    for detail in analysis.get("canonical_details") or []:
        if detail.get("technical_valid") is not True:
            errors.append(f"{detail.get('plan_row_id')}: technical audit failed")
        if detail.get("recorder_script_version") != "episode_recorder_v7":
            errors.append(f"{detail.get('plan_row_id')}: recorder is not v7")
    for detail in analysis.get("clean_anchor_details") or []:
        if detail.get("technical_valid") is not True:
            errors.append(f"{detail.get('plan_row_id')}: clean anchor audit failed")
        if detail.get("recorder_script_version") != "episode_recorder_v7":
            errors.append(f"{detail.get('plan_row_id')}: clean anchor recorder is not v7")
    return errors
