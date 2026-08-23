from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

DAY22_COLLECTION_PLAN_SHA256 = "93345b1fd8330fa9e6076b95de018e423750788d91ccb26ac15258e92916e76d"
EXPECTED_PAIR_GROUPS = tuple(f"rcv2_g{i:02d}" for i in range(1, 16))
TARGET_REPEAT_GROUPS = ("rcv2_g01", "rcv2_g04", "rcv2_g07", "rcv2_g10", "rcv2_g13")
TARGET_DIRECTION = "follower_forward"
TARGET_MAGNITUDE_MM = 40.0

SCENE_SETUP_ID = "red_cube_fixed_target_scene_v1"
OBJECT_IDENTITY = "red_cube_no_emboss_v1"
NOMINAL_START_CONDITION = "nominal_start_marker_v1"
CAMERA_SETUP = "front_wrist_v1"
ACQUISITION_CONFIGURATION = "so101_recorder_v7_60s_15hz_v1"

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

PLAN_FIELDS = (
    "schema_version",
    "day24_sequence",
    "plan_row_id",
    "pair_group_id",
    "day22_slot_index",
    "slot_role",
    "planned_physical_cause",
    "planned_intervention_type",
    "repeat_slot",
    "expected_task_outcome",
    "parameter_direction",
    "parameter_value",
    "parameter_unit",
    "collection_status",
)

RECORD_FIELDS = (
    "schema_version",
    "plan_row_id",
    "pair_group_id",
    "attempt_index",
    "episode_id",
    "raw_episode_relpath",
    "recorder_script_version",
    "technical_valid",
    "recorder_overall_pass",
    "failed_checks",
    "task_success",
    "intervention_applied",
    "single_primary_intervention",
    "changed_factor_observable",
    "parameter_direction",
    "parameter_value",
    "parameter_unit",
    "scene_setup_id",
    "object_identity",
    "nominal_start_condition",
    "camera_setup",
    "acquisition_configuration",
    "safety_abort",
    "hardware_fault",
    "experimental_valid",
    "selected_canonical",
    "exclusion_reason",
    "operator_notes",
)


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_day22_plan(path: Path) -> None:
    actual = file_sha256(path)
    if actual != DAY22_COLLECTION_PLAN_SHA256:
        raise ValueError(
            "Day22 frozen collection plan SHA256 mismatch: "
            f"expected={DAY22_COLLECTION_PLAN_SHA256} actual={actual}"
        )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def expected_day24_rows_from_day22(day22_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    sequence = 1
    for pair_index in range(1, 16):
        group = f"rcv2_g{pair_index:02d}"
        members = [row for row in day22_rows if row["pair_group_id"] == group]

        clean = [row for row in members if row["slot_index"] == "1"]
        target_primary = [row for row in members if row["slot_index"] == "2"]
        target_repeat = [
            row for row in members
            if row["slot_index"] == "5"
            and row["planned_physical_cause"] == "target_offset_or_perception"
        ]

        if len(clean) != 1 or len(target_primary) != 1:
            raise ValueError(f"Day22 group {group} missing clean or target-primary slot")

        source_rows = clean + target_primary + target_repeat
        for src in source_rows:
            is_clean = src["slot_role"] == "clean_control"
            selected.append({
                "schema_version": "evidencemm_day24_target_collection_plan_v1",
                "day24_sequence": str(sequence),
                "plan_row_id": src["plan_row_id"],
                "pair_group_id": group,
                "day22_slot_index": src["slot_index"],
                "slot_role": src["slot_role"],
                "planned_physical_cause": src["planned_physical_cause"],
                "planned_intervention_type": src["planned_intervention_type"],
                "repeat_slot": str(src["repeat_slot"]).lower(),
                "expected_task_outcome": "success" if is_clean else "failure",
                "parameter_direction": "" if is_clean else TARGET_DIRECTION,
                "parameter_value": "" if is_clean else f"{TARGET_MAGNITUDE_MM:g}",
                "parameter_unit": "" if is_clean else "mm",
                "collection_status": "pending",
            })
            sequence += 1
    return selected


def validate_day24_plan_shape(rows: list[dict[str, str]]) -> None:
    if len(rows) != 35:
        raise ValueError(f"Day24 plan must contain 35 rows, got {len(rows)}")

    groups = {row["pair_group_id"] for row in rows}
    if groups != set(EXPECTED_PAIR_GROUPS):
        raise ValueError("Day24 plan pair-group set mismatch")

    clean = [row for row in rows if row["slot_role"] == "clean_control"]
    target = [
        row for row in rows
        if row["planned_physical_cause"] == "target_offset_or_perception"
    ]
    repeat = [row for row in target if parse_bool(row["repeat_slot"])]

    if len(clean) != 15:
        raise ValueError(f"expected 15 clean rows, got {len(clean)}")
    if len(target) != 20:
        raise ValueError(f"expected 20 target rows, got {len(target)}")
    if {row["pair_group_id"] for row in repeat} != set(TARGET_REPEAT_GROUPS):
        raise ValueError("target repeat groups mismatch")

    for row in target:
        if row["parameter_direction"] != TARGET_DIRECTION:
            raise ValueError(f"{row['plan_row_id']} target direction drift")
        if float(row["parameter_value"]) != TARGET_MAGNITUDE_MM:
            raise ValueError(f"{row['plan_row_id']} target magnitude drift")
        if row["parameter_unit"] != "mm":
            raise ValueError(f"{row['plan_row_id']} target unit drift")


@dataclass(frozen=True)
class TechnicalAudit:
    episode_id: str
    raw_episode_relpath: str
    recorder_script_version: str
    technical_valid: bool
    recorder_overall_pass: bool
    failed_checks: list[str]
    sample_count: int
    csv_row_count: int
    front_image_count: int
    wrist_image_count: int
    duration_seconds: float | None
    max_tracking_error: float | None


def _image_count(path: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(
        1 for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in suffixes
    )


def _finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite sample value: {value!r}")
    return parsed


def audit_episode(episode_dir: Path) -> TechnicalAudit:
    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.csv"
    front_dir = episode_dir / "front"
    wrist_dir = episode_dir / "wrist"

    missing = [
        str(path.name)
        for path in (metadata_path, samples_path, front_dir, wrist_dir)
        if not path.exists()
    ]
    if missing:
        return TechnicalAudit(
            episode_id=episode_dir.name,
            raw_episode_relpath=episode_dir.name,
            recorder_script_version="",
            technical_valid=False,
            recorder_overall_pass=False,
            failed_checks=[f"missing:{item}" for item in missing],
            sample_count=0,
            csv_row_count=0,
            front_image_count=0,
            wrist_image_count=0,
            duration_seconds=None,
            max_tracking_error=None,
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with samples_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    required = {"frame_index", "elapsed_ns"}
    for prefix in ("observation", "action", "tracking_error"):
        required.update(f"{prefix}_{joint}" for joint in JOINT_ORDER)

    failed: list[str] = []
    if not required.issubset(fields):
        failed.append("samples_required_columns")

    if len(rows) != 900:
        failed.append("csv_row_count_exact")

    try:
        frame_indices = [int(row["frame_index"]) for row in rows]
        if frame_indices != list(range(len(rows))):
            failed.append("frame_indices_contiguous")
    except Exception:
        failed.append("frame_indices_parse")

    duration_seconds = None
    max_tracking = None
    if rows and required.issubset(fields):
        try:
            elapsed = [int(row["elapsed_ns"]) for row in rows]
            if any(b <= a for a, b in zip(elapsed, elapsed[1:])):
                failed.append("elapsed_ns_strictly_monotonic")
            duration_seconds = (
                (elapsed[-1] - elapsed[0]) / 1e9 if len(elapsed) >= 2 else 0.0
            )

            tracking_values: list[float] = []
            for row in rows:
                for prefix in ("observation", "action", "tracking_error"):
                    for joint in JOINT_ORDER:
                        value = _finite(row[f"{prefix}_{joint}"])
                        if prefix == "tracking_error":
                            tracking_values.append(value)
            if tracking_values:
                max_tracking = max(tracking_values)
        except Exception:
            failed.append("nonfinite_or_invalid_samples")

    front_count = _image_count(front_dir)
    wrist_count = _image_count(wrist_dir)
    if front_count != 900:
        failed.append("front_image_count_exact")
    if wrist_count != 900:
        failed.append("wrist_image_count_exact")

    recorder_checks = metadata.get("checks")
    if isinstance(recorder_checks, dict):
        failed.extend(
            f"recorder:{name}"
            for name, passed in recorder_checks.items()
            if passed is not True
        )
    else:
        failed.append("metadata_checks_missing")

    overall_pass = metadata.get("overall_pass") is True
    if not overall_pass:
        failed.append("recorder_overall_pass")

    result_count = (metadata.get("results") or {}).get("sample_count")
    if result_count != 900:
        failed.append("metadata_sample_count_exact")

    failed = sorted(set(failed))
    return TechnicalAudit(
        episode_id=episode_dir.name,
        raw_episode_relpath=episode_dir.name,
        recorder_script_version=str(metadata.get("script_version") or ""),
        technical_valid=not failed,
        recorder_overall_pass=overall_pass,
        failed_checks=failed,
        sample_count=int(result_count or 0),
        csv_row_count=len(rows),
        front_image_count=front_count,
        wrist_image_count=wrist_count,
        duration_seconds=duration_seconds,
        max_tracking_error=max_tracking,
    )


def next_attempt_index(records: list[dict[str, str]], plan_row_id: str) -> int:
    existing = [
        int(row["attempt_index"])
        for row in records
        if row["plan_row_id"] == plan_row_id and row.get("attempt_index")
    ]
    return max(existing, default=0) + 1


def selected_for_plan(records: list[dict[str, str]], plan_row_id: str) -> list[dict[str, str]]:
    return [
        row for row in records
        if row["plan_row_id"] == plan_row_id
        and parse_bool(row.get("selected_canonical"))
    ]


def evaluate_experimental_validity(
    *,
    plan_row: dict[str, str],
    technical_valid: bool,
    task_success: bool,
    intervention_applied: bool,
    single_primary_intervention: bool,
    changed_factor_observable: bool,
    safety_abort: bool,
    hardware_fault: bool,
) -> tuple[bool, str]:
    if not technical_valid:
        return False, "technical_exclusion"
    if safety_abort:
        return False, "safety_abort"
    if hardware_fault:
        return False, "hardware_fault"

    if plan_row["slot_role"] == "clean_control":
        if intervention_applied:
            return False, "clean_has_intervention"
        if not task_success:
            return False, "clean_task_failure"
        return True, ""

    if plan_row["planned_physical_cause"] != "target_offset_or_perception":
        return False, "unexpected_day24_cause"
    if not intervention_applied:
        return False, "declared_target_intervention_not_applied"
    if not single_primary_intervention:
        return False, "multiple_primary_interventions"
    if not changed_factor_observable:
        return False, "changed_factor_not_observable"
    if task_success:
        return False, "target_intervention_did_not_induce_failure"
    return True, ""


def build_record(
    *,
    plan_row: dict[str, str],
    audit: TechnicalAudit,
    attempt_index: int,
    task_success: bool,
    intervention_applied: bool,
    single_primary_intervention: bool,
    changed_factor_observable: bool,
    safety_abort: bool,
    hardware_fault: bool,
    selected_canonical: bool,
    operator_notes: str,
) -> dict[str, str]:
    experimental_valid, exclusion_reason = evaluate_experimental_validity(
        plan_row=plan_row,
        technical_valid=audit.technical_valid,
        task_success=task_success,
        intervention_applied=intervention_applied,
        single_primary_intervention=single_primary_intervention,
        changed_factor_observable=changed_factor_observable,
        safety_abort=safety_abort,
        hardware_fault=hardware_fault,
    )

    is_clean = plan_row["slot_role"] == "clean_control"
    return {
        "schema_version": "evidencemm_day24_target_collection_record_v1",
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
        "parameter_direction": "" if is_clean else TARGET_DIRECTION,
        "parameter_value": "" if is_clean else f"{TARGET_MAGNITUDE_MM:g}",
        "parameter_unit": "" if is_clean else "mm",
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


def canonical_records(plan_rows: list[dict[str, str]], records: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in records if parse_bool(row.get("selected_canonical"))]
    by_plan: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_plan.setdefault(row["plan_row_id"], []).append(row)

    duplicates = {key: rows for key, rows in by_plan.items() if len(rows) > 1}
    if duplicates:
        raise ValueError(
            "more than one canonical record for plan rows: "
            + ", ".join(sorted(duplicates))
        )

    plan_ids = {row["plan_row_id"] for row in plan_rows}
    if any(row["plan_row_id"] not in plan_ids for row in selected):
        raise ValueError("records contain unknown plan_row_id")
    return selected


def analyze_collection(
    *,
    plan_rows: list[dict[str, str]],
    records: list[dict[str, str]],
    raw_root: Path | None = None,
) -> dict[str, Any]:
    validate_day24_plan_shape(plan_rows)
    canonical = canonical_records(plan_rows, records)

    plan_by_id = {row["plan_row_id"]: row for row in plan_rows}
    canonical_by_id = {row["plan_row_id"]: row for row in canonical}

    clean = [
        row for row in canonical
        if plan_by_id[row["plan_row_id"]]["slot_role"] == "clean_control"
    ]
    target = [
        row for row in canonical
        if plan_by_id[row["plan_row_id"]]["planned_physical_cause"]
        == "target_offset_or_perception"
    ]

    per_group = {}
    for group in EXPECTED_PAIR_GROUPS:
        group_plan = [row for row in plan_rows if row["pair_group_id"] == group]
        group_selected = [
            canonical_by_id[row["plan_row_id"]]
            for row in group_plan
            if row["plan_row_id"] in canonical_by_id
        ]
        per_group[group] = {
            "expected_slots": len(group_plan),
            "canonical_slots": len(group_selected),
            "complete": len(group_selected) == len(group_plan),
        }

    technical_exclusions = sum(
        1 for row in records if not parse_bool(row.get("technical_valid"))
    )
    experimental_exclusions = sum(
        1
        for row in records
        if parse_bool(row.get("technical_valid"))
        and not parse_bool(row.get("experimental_valid"))
    )

    canonical_details: list[dict[str, Any]] = []
    if raw_root is not None:
        for row in canonical:
            audit = audit_episode(raw_root / row["raw_episode_relpath"])
            canonical_details.append({
                "plan_row_id": row["plan_row_id"],
                "pair_group_id": row["pair_group_id"],
                "episode_id": row["episode_id"],
                "task_success": parse_bool(row["task_success"]),
                "technical_valid": audit.technical_valid,
                "duration_seconds": (
                    None if audit.duration_seconds is None
                    else round(audit.duration_seconds, 6)
                ),
                "max_tracking_error": (
                    None if audit.max_tracking_error is None
                    else round(audit.max_tracking_error, 4)
                ),
                "recorder_script_version": audit.recorder_script_version,
            })

    return {
        "schema_version": "evidencemm_day24_target_collection_analysis_v1",
        "status": (
            "complete"
            if len(canonical) == 35
            else "in_progress"
        ),
        "attempt_count": len(records),
        "canonical_episode_count": len(canonical),
        "expected_canonical_episode_count": 35,
        "clean_canonical_count": len(clean),
        "clean_success_count": sum(parse_bool(row["task_success"]) for row in clean),
        "target_canonical_count": len(target),
        "target_failure_count": sum(not parse_bool(row["task_success"]) for row in target),
        "technical_exclusion_attempt_count": technical_exclusions,
        "experimental_exclusion_attempt_count": experimental_exclusions,
        "pair_group_count": 15,
        "complete_pair_group_count": sum(item["complete"] for item in per_group.values()),
        "target_parameter": {
            "direction": TARGET_DIRECTION,
            "magnitude": TARGET_MAGNITUDE_MM,
            "unit": "mm",
        },
        "future_split_materialized": False,
        "per_group": per_group,
        "canonical_details": canonical_details,
    }


def validate_final_analysis(analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "canonical_episode_count": 35,
        "clean_canonical_count": 15,
        "clean_success_count": 15,
        "target_canonical_count": 20,
        "target_failure_count": 20,
        "complete_pair_group_count": 15,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            errors.append(f"{key}: expected {value}, got {analysis.get(key)}")

    if analysis.get("future_split_materialized") is not False:
        errors.append("future split must not be materialized on Day24")

    target = analysis.get("target_parameter") or {}
    if target.get("direction") != TARGET_DIRECTION:
        errors.append("target direction mismatch")
    if float(target.get("magnitude", -1)) != TARGET_MAGNITUDE_MM:
        errors.append("target magnitude mismatch")
    if target.get("unit") != "mm":
        errors.append("target unit mismatch")

    for detail in analysis.get("canonical_details") or []:
        if detail.get("technical_valid") is not True:
            errors.append(
                f"{detail.get('plan_row_id')}: canonical episode technical audit failed"
            )
        if detail.get("recorder_script_version") != "episode_recorder_v7":
            errors.append(
                f"{detail.get('plan_row_id')}: recorder version is not episode_recorder_v7"
            )
    return errors
