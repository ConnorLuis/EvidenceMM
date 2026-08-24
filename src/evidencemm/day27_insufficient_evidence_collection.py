from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from evidencemm.day24_target_collection import (
    ACQUISITION_CONFIGURATION,
    CAMERA_SETUP,
    DAY22_COLLECTION_PLAN_SHA256,
    EXPECTED_PAIR_GROUPS,
    NOMINAL_START_CONDITION,
    OBJECT_IDENTITY,
    SCENE_SETUP_ID,
    TechnicalAudit,
    audit_episode,
    bool_text,
    file_sha256,
    parse_bool,
)

AMBIGUITY_PROTOCOL = "blinded_single_cause_challenge_v2"
PLANNED_INTERVENTION_TYPE = "ambiguity_protocol"

TARGET_MILD_VARIANT = "target_mild_20mm_forward"
GRIPPER_VARIANT = "gripper_late_30_40mm_upward_progress"
TRAJECTORY_MILD_VARIANT = "trajectory_mild_25mm_forward"
AMBIGUITY_VARIANT_ROTATION = (
    TARGET_MILD_VARIANT,
    GRIPPER_VARIANT,
    TRAJECTORY_MILD_VARIANT,
)
AMBIGUITY_VARIANT_SPECS = {
    TARGET_MILD_VARIANT: {
        "admin_physical_cause": "target_offset_or_perception",
        "admin_intervention_type": "object_target_pose_offset",
        "operator_instruction": (
            "Move the red cube 20 mm toward Follower-forward from the nominal marker; "
            "execute the original nominal grasp path and do not compensate."
        ),
    },
    GRIPPER_VARIANT: {
        "admin_physical_cause": "gripper_close_timing",
        "admin_intervention_type": "manual_gripper_close_timing_shift",
        "operator_instruction": (
            "Keep object and path nominal; close late only after 30-40 mm upward "
            "progress beyond the nominal grasp-close point."
        ),
    },
    TRAJECTORY_MILD_VARIANT: {
        "admin_physical_cause": "trajectory_execution_deviation",
        "admin_intervention_type": "manual_bounded_trajectory_deviation",
        "operator_instruction": (
            "Keep object and gripper timing nominal; near above the cube move the end "
            "effector about 25 mm Follower-forward, hold that offset while descending, "
            "and do not compensate/re-grasp."
        ),
    },
}


def ambiguity_variant_for_group(pair_group_id: str) -> str:
    try:
        index = int(pair_group_id.rsplit("g", 1)[1])
    except Exception as exc:
        raise ValueError(f"invalid pair_group_id: {pair_group_id}") from exc
    if index < 1 or index > 15:
        raise ValueError("pair_group_id must be rcv2_g01..rcv2_g15")
    return AMBIGUITY_VARIANT_ROTATION[(index - 1) % 3]


def ambiguity_protocol_value(pair_group_id: str) -> str:
    return f"{AMBIGUITY_PROTOCOL}:{ambiguity_variant_for_group(pair_group_id)}"

PLAN_FIELDS = (
    "schema_version",
    "day27_sequence",
    "plan_row_id",
    "pair_group_id",
    "day22_slot_index",
    "slot_role",
    "planned_physical_cause",
    "planned_intervention_type",
    "repeat_slot",
    "expected_task_outcome",
    "ambiguity_protocol",
    "clean_anchor_plan_row_id",
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
    "ambiguity_protocol_followed",
    "deliberate_known_cause_intervention",
    "intentional_failure_injection",
    "multiple_primary_interventions",
    "scene_comparable",
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


def write_csv_lf(
    path: Path,
    rows: list[dict[str, Any]],
    fields: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def assert_frozen_day22_plan(path: Path) -> None:
    actual = file_sha256(path)
    if actual != DAY22_COLLECTION_PLAN_SHA256:
        raise ValueError(
            "Day22 frozen collection plan SHA256 mismatch: "
            f"expected={DAY22_COLLECTION_PLAN_SHA256} actual={actual}"
        )


def expected_day27_rows_from_day22(
    day22_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for sequence, pair_index in enumerate(range(1, 16), start=1):
        group = f"rcv2_g{pair_index:02d}"
        matches = [
            row
            for row in day22_rows
            if row["pair_group_id"] == group
            and row["slot_index"] == "6"
        ]
        if len(matches) != 1:
            raise ValueError(f"{group}: expected exactly one Day22 s06 row")
        src = matches[0]
        if src["slot_role"] != "insufficient_evidence_candidate":
            raise ValueError(f"{group}: s06 role drift")
        if src["planned_physical_cause"] != "unknown":
            raise ValueError(f"{group}: s06 physical cause drift")
        if src["planned_intervention_type"] != PLANNED_INTERVENTION_TYPE:
            raise ValueError(f"{group}: s06 intervention type drift")
        if parse_bool(src["repeat_slot"]):
            raise ValueError(f"{group}: s06 cannot be repeat slot")
        selected.append(
            {
                "schema_version": (
                    "evidencemm_day27_insufficient_evidence_collection_plan_v2"
                ),
                "day27_sequence": str(sequence),
                "plan_row_id": src["plan_row_id"],
                "pair_group_id": group,
                "day22_slot_index": "6",
                "slot_role": src["slot_role"],
                "planned_physical_cause": src["planned_physical_cause"],
                "planned_intervention_type": src["planned_intervention_type"],
                "repeat_slot": "false",
                "expected_task_outcome": "failure",
                "ambiguity_protocol": ambiguity_protocol_value(group),
                "clean_anchor_plan_row_id": f"{group}_s01",
                "collection_status": "pending",
            }
        )
    return selected


def validate_day27_plan_shape(rows: list[dict[str, str]]) -> None:
    if len(rows) != 15:
        raise ValueError(f"Day27 plan must contain 15 rows, got {len(rows)}")
    if {row["pair_group_id"] for row in rows} != set(EXPECTED_PAIR_GROUPS):
        raise ValueError("Day27 pair-group set mismatch")
    if len({row["plan_row_id"] for row in rows}) != 15:
        raise ValueError("Day27 plan_row_id values must be unique")

    for row in rows:
        group = row["pair_group_id"]
        if row["plan_row_id"] != f"{group}_s06":
            raise ValueError(f"{group}: Day27 row must be s06")
        if row["day22_slot_index"] != "6":
            raise ValueError(f"{group}: Day22 slot index drift")
        if row["slot_role"] != "insufficient_evidence_candidate":
            raise ValueError(f"{group}: role drift")
        if row["planned_physical_cause"] != "unknown":
            raise ValueError(f"{group}: planned physical cause must remain unknown")
        if row["planned_intervention_type"] != PLANNED_INTERVENTION_TYPE:
            raise ValueError(f"{group}: intervention type drift")
        if parse_bool(row["repeat_slot"]):
            raise ValueError(f"{group}: insufficient candidate cannot be repeat")
        if row["expected_task_outcome"] != "failure":
            raise ValueError(f"{group}: s06 requires a task failure")
        expected_protocol = ambiguity_protocol_value(group)
        if row["ambiguity_protocol"] != expected_protocol:
            raise ValueError(f"{group}: ambiguity protocol drift")

    variants = [row["ambiguity_protocol"].split(":", 1)[1] for row in rows]
    for variant in AMBIGUITY_VARIANT_ROTATION:
        if variants.count(variant) != 5:
            raise ValueError(f"ambiguity variant count drift for {variant}")
        if row["clean_anchor_plan_row_id"] != f"{group}_s01":
            raise ValueError(f"{group}: clean anchor drift")


def selected_day24_clean_anchors(
    day24_plan_rows: list[dict[str, str]],
    day24_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    clean_plan_ids = {
        row["plan_row_id"]
        for row in day24_plan_rows
        if row["slot_role"] == "clean_control"
    }
    selected = [
        row
        for row in day24_records
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
            raise ValueError(f"{group}: expected one selected clean anchor")
        row = rows[0]
        if row["plan_row_id"] != f"{group}_s01":
            raise ValueError(f"{group}: clean anchor must be s01")
        if not parse_bool(row.get("technical_valid")):
            raise ValueError(f"{group}: clean anchor technical invalid")
        if not parse_bool(row.get("experimental_valid")):
            raise ValueError(f"{group}: clean anchor experimental invalid")
        if not parse_bool(row.get("task_success")):
            raise ValueError(f"{group}: clean anchor must be success")
        if parse_bool(row.get("intervention_applied")):
            raise ValueError(f"{group}: clean anchor must have no intervention")
    return selected


def _selected(
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in records
        if parse_bool(row.get("selected_canonical"))
    ]


def validate_preexisting_final_slots(
    *,
    day24_plan_rows: list[dict[str, str]],
    day24_records: list[dict[str, str]],
    day25_plan_rows: list[dict[str, str]],
    day25_records: list[dict[str, str]],
    day26_plan_rows: list[dict[str, str]],
    day26_records: list[dict[str, str]],
) -> dict[str, Any]:
    anchors = selected_day24_clean_anchors(day24_plan_rows, day24_records)

    def valid_selected(records: list[dict[str, str]], expected_count: int, label: str):
        rows = _selected(records)
        if len(rows) != expected_count:
            raise ValueError(
                f"{label}: expected {expected_count} selected canonical rows, got {len(rows)}"
            )
        for row in rows:
            if not parse_bool(row.get("technical_valid")):
                raise ValueError(f"{label}:{row['plan_row_id']}: technical invalid")
            if not parse_bool(row.get("experimental_valid")):
                raise ValueError(f"{label}:{row['plan_row_id']}: experimental invalid")
        return rows

    day24_selected = valid_selected(day24_records, 35, "Day24")
    target = [
        row for row in day24_selected
        if row["plan_row_id"] not in {anchor["plan_row_id"] for anchor in anchors}
    ]
    if len(target) != 20:
        raise ValueError(f"Day24 target canonical count must be 20, got {len(target)}")
    if any(parse_bool(row.get("task_success")) for row in target):
        raise ValueError("Day24 target canonical rows must be failures")

    gripper = valid_selected(day25_records, 20, "Day25")
    if any(parse_bool(row.get("task_success")) for row in gripper):
        raise ValueError("Day25 gripper canonical rows must be failures")

    trajectory = valid_selected(day26_records, 20, "Day26")
    if any(parse_bool(row.get("task_success")) for row in trajectory):
        raise ValueError("Day26 trajectory canonical rows must be failures")

    controlled = target + gripper + trajectory
    by_group = {group: 0 for group in EXPECTED_PAIR_GROUPS}
    for row in controlled:
        by_group[row["pair_group_id"]] += 1
    if any(count != 4 for count in by_group.values()):
        bad = {g: n for g, n in by_group.items() if n != 4}
        raise ValueError(f"preexisting controlled slots per group mismatch: {bad}")

    return {
        "clean_anchors": anchors,
        "controlled_records": controlled,
        "controlled_count": len(controlled),
        "controlled_by_group": by_group,
        "preexisting_recollection_required": False,
        "new_clean_collection_required": False,
    }


def evaluate_experimental_validity(
    *,
    plan_row: dict[str, str],
    technical_valid: bool,
    task_success: bool,
    ambiguity_protocol_followed: bool,
    deliberate_known_cause_intervention: bool,
    intentional_failure_injection: bool,
    multiple_primary_interventions: bool,
    scene_comparable: bool,
    safety_abort: bool,
    hardware_fault: bool,
) -> tuple[bool, str]:
    if not technical_valid:
        return False, "technical_exclusion"
    if safety_abort:
        return False, "safety_abort"
    if hardware_fault:
        return False, "hardware_fault"
    if plan_row["slot_role"] != "insufficient_evidence_candidate":
        return False, "unexpected_day27_slot_role"
    if plan_row["planned_physical_cause"] != "unknown":
        return False, "unexpected_day27_physical_cause"
    if plan_row["planned_intervention_type"] != PLANNED_INTERVENTION_TYPE:
        return False, "unexpected_day27_intervention_type"
    if not ambiguity_protocol_followed:
        return False, "ambiguity_protocol_not_followed"
    # Day22 allows the physical cause to be administratively known while
    # answerability is judged only from model-visible evidence. Day27 v2 uses
    # exactly one predeclared single-cause challenge per s06 candidate.
    if not deliberate_known_cause_intervention:
        return False, "assigned_ambiguity_challenge_not_applied"
    if not intentional_failure_injection:
        return False, "predeclared_failure_challenge_not_applied"
    if multiple_primary_interventions:
        return False, "multiple_primary_interventions"
    if not scene_comparable:
        return False, "pair_group_setup_not_comparable"
    if task_success:
        return False, "ambiguity_candidate_task_success"
    return True, ""


def build_record(
    *,
    plan_row: dict[str, str],
    audit: TechnicalAudit,
    attempt_index: int,
    task_success: bool,
    ambiguity_protocol_followed: bool,
    deliberate_known_cause_intervention: bool,
    intentional_failure_injection: bool,
    multiple_primary_interventions: bool,
    scene_comparable: bool,
    safety_abort: bool,
    hardware_fault: bool,
    selected_canonical: bool,
    operator_notes: str,
) -> dict[str, str]:
    experimental_valid, exclusion_reason = evaluate_experimental_validity(
        plan_row=plan_row,
        technical_valid=audit.technical_valid,
        task_success=task_success,
        ambiguity_protocol_followed=ambiguity_protocol_followed,
        deliberate_known_cause_intervention=deliberate_known_cause_intervention,
        intentional_failure_injection=intentional_failure_injection,
        multiple_primary_interventions=multiple_primary_interventions,
        scene_comparable=scene_comparable,
        safety_abort=safety_abort,
        hardware_fault=hardware_fault,
    )
    return {
        "schema_version": (
            "evidencemm_day27_insufficient_evidence_collection_record_v1"
        ),
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
        "ambiguity_protocol_followed": bool_text(ambiguity_protocol_followed),
        "deliberate_known_cause_intervention": bool_text(
            deliberate_known_cause_intervention
        ),
        "intentional_failure_injection": bool_text(intentional_failure_injection),
        "multiple_primary_interventions": bool_text(multiple_primary_interventions),
        "scene_comparable": bool_text(scene_comparable),
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


def canonical_records(
    plan_rows: list[dict[str, str]],
    records: list[dict[str, str]],
) -> list[dict[str, str]]:
    plan_ids = {row["plan_row_id"] for row in plan_rows}
    selected = [
        row for row in records if parse_bool(row.get("selected_canonical"))
    ]
    by_plan: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        if row["plan_row_id"] not in plan_ids:
            raise ValueError("Day27 records contain unknown plan_row_id")
        by_plan.setdefault(row["plan_row_id"], []).append(row)
    duplicates = [key for key, values in by_plan.items() if len(values) > 1]
    if duplicates:
        raise ValueError(
            "more than one Day27 canonical record for: "
            + ", ".join(sorted(duplicates))
        )
    return selected


def analyze_collection(
    *,
    plan_rows: list[dict[str, str]],
    records: list[dict[str, str]],
    preexisting: dict[str, Any],
    raw_root: Path | None = None,
) -> dict[str, Any]:
    validate_day27_plan_shape(plan_rows)
    canonical = canonical_records(plan_rows, records)
    canonical_by_id = {row["plan_row_id"]: row for row in canonical}
    anchors = list(preexisting["clean_anchors"])
    anchor_by_group = {row["pair_group_id"]: row for row in anchors}
    controlled_by_group = dict(preexisting["controlled_by_group"])

    per_group: dict[str, Any] = {}
    for group in EXPECTED_PAIR_GROUPS:
        has_candidate = f"{group}_s06" in canonical_by_id
        preexisting_ready = (
            group in anchor_by_group
            and controlled_by_group.get(group) == 4
        )
        per_group[group] = {
            "clean_anchor_present": group in anchor_by_group,
            "controlled_canonical_slots": controlled_by_group.get(group, 0),
            "insufficient_candidate_present": has_candidate,
            "complete_through_day27": preexisting_ready and has_candidate,
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
    noncanonical = [
        row for row in records if not parse_bool(row.get("selected_canonical"))
    ]
    success_attempts = sum(
        1
        for row in records
        if parse_bool(row.get("technical_valid"))
        and parse_bool(row.get("task_success"))
    )

    canonical_details: list[dict[str, Any]] = []
    clean_anchor_details: list[dict[str, Any]] = []
    if raw_root is not None:
        for row in canonical:
            audit = audit_episode(raw_root / row["raw_episode_relpath"])
            canonical_details.append(
                {
                    "plan_row_id": row["plan_row_id"],
                    "pair_group_id": row["pair_group_id"],
                    "episode_id": row["episode_id"],
                    "task_success": parse_bool(row["task_success"]),
                    "technical_valid": audit.technical_valid,
                    "duration_seconds": (
                        None
                        if audit.duration_seconds is None
                        else round(audit.duration_seconds, 6)
                    ),
                    "max_tracking_error": (
                        None
                        if audit.max_tracking_error is None
                        else round(audit.max_tracking_error, 4)
                    ),
                    "recorder_script_version": audit.recorder_script_version,
                }
            )
        for row in anchors:
            audit = audit_episode(raw_root / row["raw_episode_relpath"])
            clean_anchor_details.append(
                {
                    "plan_row_id": row["plan_row_id"],
                    "pair_group_id": row["pair_group_id"],
                    "episode_id": row["episode_id"],
                    "technical_valid": audit.technical_valid,
                    "recorder_script_version": audit.recorder_script_version,
                }
            )

    eligible_count = len(anchors) + int(preexisting["controlled_count"]) + len(canonical)
    return {
        "schema_version": (
            "evidencemm_day27_insufficient_evidence_collection_analysis_v1"
        ),
        "status": "complete" if len(canonical) == 15 else "in_progress",
        "new_attempt_count": len(records),
        "new_insufficient_candidate_canonical_count": len(canonical),
        "expected_new_insufficient_candidate_canonical_count": 15,
        "insufficient_candidate_failure_count": sum(
            not parse_bool(row["task_success"]) for row in canonical
        ),
        "successful_noncanonical_attempt_count": success_attempts,
        "recollection_attempt_count": len(noncanonical),
        "technical_exclusion_attempt_count": technical_exclusions,
        "experimental_exclusion_attempt_count": experimental_exclusions,
        "clean_anchor_count": len(anchors),
        "clean_anchor_success_count": sum(
            parse_bool(row["task_success"]) for row in anchors
        ),
        "controlled_canonical_count": int(preexisting["controlled_count"]),
        "eligible_target_episode_count_through_day27": eligible_count,
        "pair_group_count": 15,
        "complete_pair_group_count": sum(
            entry["complete_through_day27"] for entry in per_group.values()
        ),
        "ambiguity_protocol": {
            "name": AMBIGUITY_PROTOCOL,
            "planned_intervention_type": PLANNED_INTERVENTION_TYPE,
            "planned_physical_cause_remains_unknown": True,
            "single_admin_known_cause_challenge_required": True,
            "technical_corruption_allowed": False,
            "multiple_primary_interventions_allowed": False,
            "task_failure_required_for_canonical_candidate": True,
            "answerability_prejudged_on_day27": False,
            "human_review_blind_to_admin_cause_until_answerability_judged": True,
            "variant_counts": {
                variant: sum(
                    1
                    for row in plan_rows
                    if row["ambiguity_protocol"] == f"{AMBIGUITY_PROTOCOL}:{variant}"
                )
                for variant in AMBIGUITY_VARIANT_ROTATION
            },
        },
        "preexisting_recollection_required": bool(
            preexisting["preexisting_recollection_required"]
        ),
        "new_clean_collection_required": bool(
            preexisting["new_clean_collection_required"]
        ),
        "future_split_materialized": False,
        "raw_attempt_retention_completeness_asserted": False,
        "per_group": per_group,
        "canonical_details": canonical_details,
        "clean_anchor_details": clean_anchor_details,
    }


def validate_final_analysis(analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "new_insufficient_candidate_canonical_count": 15,
        "insufficient_candidate_failure_count": 15,
        "clean_anchor_count": 15,
        "clean_anchor_success_count": 15,
        "controlled_canonical_count": 60,
        "eligible_target_episode_count_through_day27": 90,
        "complete_pair_group_count": 15,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            errors.append(
                f"{key}: expected {value}, got {analysis.get(key)}"
            )
    if analysis.get("preexisting_recollection_required") is not False:
        errors.append("Day24-26 frozen slots must not require recollection")
    if analysis.get("new_clean_collection_required") is not False:
        errors.append("Day27 must reuse the 15 Day24 clean anchors")
    if analysis.get("future_split_materialized") is not False:
        errors.append("future split must remain unmaterialized until Day30")

    protocol = analysis.get("ambiguity_protocol") or {}
    if protocol.get("name") != AMBIGUITY_PROTOCOL:
        errors.append("ambiguity protocol mismatch")
    if protocol.get("single_admin_known_cause_challenge_required") is not True:
        errors.append("Day27 v2 requires one admin-known single-cause challenge")
    if protocol.get("technical_corruption_allowed") is not False:
        errors.append("technical corruption must be forbidden in s06")
    if protocol.get("multiple_primary_interventions_allowed") is not False:
        errors.append("multiple primary interventions must be forbidden in s06")
    if protocol.get("answerability_prejudged_on_day27") is not False:
        errors.append("Day27 must not pre-judge evidence answerability")
    counts = protocol.get("variant_counts") or {}
    for variant in AMBIGUITY_VARIANT_ROTATION:
        if counts.get(variant) != 5:
            errors.append(f"ambiguity variant count must be 5 for {variant}")

    for detail in analysis.get("canonical_details") or []:
        if detail.get("technical_valid") is not True:
            errors.append(f"{detail.get('plan_row_id')}: technical audit failed")
        if detail.get("task_success") is not False:
            errors.append(f"{detail.get('plan_row_id')}: canonical s06 must fail")
        if detail.get("recorder_script_version") != "episode_recorder_v7":
            errors.append(
                f"{detail.get('plan_row_id')}: recorder is not episode_recorder_v7"
            )
    for detail in analysis.get("clean_anchor_details") or []:
        if detail.get("technical_valid") is not True:
            errors.append(
                f"{detail.get('plan_row_id')}: clean anchor technical audit failed"
            )
        if detail.get("recorder_script_version") != "episode_recorder_v7":
            errors.append(
                f"{detail.get('plan_row_id')}: clean anchor recorder is not v7"
            )
    return errors
