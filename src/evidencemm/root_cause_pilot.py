\
from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.state_action_selection import JOINT_ORDER, load_state_action_samples


CONFIG_SCHEMA = "evidencemm_day23_root_cause_pilot_config_v1"
PLAN_SCHEMA = "evidencemm_day23_pilot_plan_v1"
RECORD_SCHEMA = "evidencemm_day23_pilot_record_v1"
ANALYSIS_SCHEMA = "evidencemm_day23_pilot_analysis_v1"
FREEZE_SCHEMA = "evidencemm_day23_intervention_parameters_v1"
FREEZE_STATUS = "excluded_pilot_passed_intervention_parameters_frozen"

CAUSES = (
    "target_offset_or_perception",
    "gripper_close_timing",
    "trajectory_execution_deviation",
)

RECORD_COLUMNS = (
    "schema_version",
    "pilot_row_id",
    "pilot_group_id",
    "sequence_order",
    "intensity_rank",
    "intensity_label",
    "pilot_role",
    "planned_physical_cause",
    "planned_intervention_type",
    "episode_id",
    "raw_episode_relpath",
    "recorder_overall_pass",
    "failed_checks",
    "task_success",
    "intervention_predeclared",
    "intervention_applied",
    "single_primary_intervention",
    "parameter_direction",
    "parameter_value",
    "parameter_unit",
    "changed_factor_observable",
    "observable_modalities",
    "gripper_transition_verified_as_grasp_close",
    "safety_abort",
    "hardware_fault",
    "operator_notes",
)

TRUE_VALUES = {"true", "1", "yes", "y"}
FALSE_VALUES = {"false", "0", "no", "n"}


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{field} must be boolean-like, got {value!r}")


def _parse_optional_bool(value: str, *, field: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized == "":
        return None
    return _parse_bool(normalized, field=field)


def _parse_optional_float(value: str, *, field: str) -> float | None:
    normalized = str(value).strip()
    if normalized == "":
        return None
    number = float(normalized)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def split_semicolon(value: str) -> tuple[str, ...]:
    return tuple(
        token.strip()
        for token in str(value).split(";")
        if token.strip()
    )


class PilotPlanRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    pilot_row_id: str
    pilot_group_id: str
    sequence_order: int
    intensity_rank: int
    intensity_label: str
    pilot_role: str
    planned_physical_cause: str
    planned_intervention_type: str
    final_benchmark_eligible: bool
    operation_summary: str

    @model_validator(mode="after")
    def validate_contract(self):
        if self.schema_version != PLAN_SCHEMA:
            raise ValueError("unexpected Day23 pilot plan schema")
        if self.final_benchmark_eligible:
            raise ValueError("Day23 pilot can never be final benchmark eligible")
        if self.pilot_role == "clean_control":
            if self.intensity_rank != 0:
                raise ValueError("clean control intensity rank must be 0")
            if self.planned_physical_cause != "none_clean":
                raise ValueError("clean control cause must be none_clean")
        else:
            if self.intensity_rank not in {1, 2, 3}:
                raise ValueError("intervention intensity rank must be 1..3")
            if self.planned_physical_cause not in CAUSES:
                raise ValueError("unsupported pilot physical cause")
        return self


class PilotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    pilot_row_id: str
    pilot_group_id: str
    sequence_order: int
    intensity_rank: int
    intensity_label: str
    pilot_role: str
    planned_physical_cause: str
    planned_intervention_type: str

    episode_id: str
    raw_episode_relpath: str
    recorder_overall_pass: bool
    failed_checks: tuple[str, ...]
    task_success: bool
    intervention_predeclared: bool
    intervention_applied: bool
    single_primary_intervention: bool
    parameter_direction: str
    parameter_value: float | None
    parameter_unit: str
    changed_factor_observable: bool
    observable_modalities: tuple[str, ...]
    gripper_transition_verified_as_grasp_close: bool | None
    safety_abort: bool
    hardware_fault: bool
    operator_notes: str

    @model_validator(mode="after")
    def validate_contract(self):
        if self.schema_version != RECORD_SCHEMA:
            raise ValueError("unexpected Day23 pilot record schema")
        if not self.episode_id:
            raise ValueError(f"{self.pilot_row_id}: episode_id is required")
        if not self.raw_episode_relpath:
            raise ValueError(f"{self.pilot_row_id}: raw_episode_relpath is required")
        if self.recorder_overall_pass and self.failed_checks:
            raise ValueError(
                f"{self.pilot_row_id}: PASS recorder cannot have failed_checks"
            )
        if not self.recorder_overall_pass and not self.failed_checks:
            raise ValueError(
                f"{self.pilot_row_id}: FAIL recorder requires failed_checks"
            )
        if self.safety_abort or self.hardware_fault:
            return self

        allowed_modalities = {
            "front", "wrist", "observation", "action", "tracking_error"
        }
        if not set(self.observable_modalities).issubset(allowed_modalities):
            raise ValueError(
                f"{self.pilot_row_id}: unsupported observable modality"
            )

        if self.pilot_role == "clean_control":
            if self.intervention_predeclared or self.intervention_applied:
                raise ValueError("clean control cannot apply intervention")
            if not self.single_primary_intervention:
                raise ValueError(
                    "clean control still requires single-primary contract=true"
                )
            if self.parameter_direction or self.parameter_value is not None or self.parameter_unit:
                raise ValueError("clean control parameter fields must be blank")
        else:
            if not self.intervention_predeclared:
                raise ValueError("controlled intervention must be predeclared")
            if not self.intervention_applied:
                raise ValueError("controlled intervention must be applied")
            if not self.single_primary_intervention:
                raise ValueError("multiple primary interventions are invalid")
            if not self.parameter_direction:
                raise ValueError("controlled intervention requires direction")

            if self.planned_physical_cause in {
                "target_offset_or_perception",
                "trajectory_execution_deviation",
            }:
                if self.parameter_value is None or self.parameter_value <= 0:
                    raise ValueError(
                        "target/trajectory intervention requires positive numeric value"
                    )
                if self.parameter_unit != "mm":
                    raise ValueError("target/trajectory parameter unit must be mm")

            if self.planned_physical_cause == "gripper_close_timing":
                if self.parameter_direction not in {"early", "late"}:
                    raise ValueError("gripper direction must be early or late")
                if self.parameter_value is not None or self.parameter_unit:
                    raise ValueError(
                        "gripper numeric value is derived from samples and must be blank"
                    )
                if self.gripper_transition_verified_as_grasp_close is None:
                    raise ValueError(
                        "gripper pilot requires transition verification field"
                    )
        return self


def load_plan_csv(path: Path) -> list[PilotPlanRow]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            normalized = dict(raw)
            normalized["sequence_order"] = int(normalized["sequence_order"])
            normalized["intensity_rank"] = int(normalized["intensity_rank"])
            normalized["final_benchmark_eligible"] = _parse_bool(
                normalized["final_benchmark_eligible"],
                field="final_benchmark_eligible",
            )
            rows.append(PilotPlanRow.model_validate(normalized))
    validate_plan(rows)
    return rows


def validate_plan(rows: Sequence[PilotPlanRow]) -> None:
    if len(rows) != 12:
        raise ValueError(f"Day23 pilot plan requires 12 rows, got {len(rows)}")
    if [row.sequence_order for row in rows] != list(range(1, 13)):
        raise ValueError("Day23 sequence_order must be exactly 1..12")
    ids = [row.pilot_row_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("pilot_row_id must be unique")

    by_group = defaultdict(list)
    for row in rows:
        by_group[row.pilot_group_id].append(row)
    if sorted(by_group) != ["p23_g01", "p23_g02", "p23_g03"]:
        raise ValueError("pilot groups must be p23_g01..p23_g03")

    for index, group_id in enumerate(sorted(by_group), start=1):
        group = by_group[group_id]
        if len(group) != 4:
            raise ValueError(f"{group_id}: expected 4 rows")
        roles = Counter(row.pilot_role for row in group)
        if roles != Counter({"controlled_cause": 3, "clean_control": 1}):
            raise ValueError(f"{group_id}: invalid role composition")
        causes = {
            row.planned_physical_cause
            for row in group
            if row.pilot_role == "controlled_cause"
        }
        if causes != set(CAUSES):
            raise ValueError(f"{group_id}: must contain all three causes")
        for row in group:
            if row.pilot_role == "controlled_cause" and row.intensity_rank != index:
                raise ValueError(f"{group_id}: intensity rank must equal group rank")


def load_records_csv(path: Path) -> list[PilotRecord]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RECORD_COLUMNS:
            raise ValueError("Day23 record CSV columns differ from frozen template")
        rows = []
        for raw in reader:
            normalized = dict(raw)
            normalized["sequence_order"] = int(normalized["sequence_order"])
            normalized["intensity_rank"] = int(normalized["intensity_rank"])
            for field in (
                "recorder_overall_pass", "task_success",
                "intervention_predeclared", "intervention_applied",
                "single_primary_intervention", "changed_factor_observable",
                "safety_abort", "hardware_fault",
            ):
                normalized[field] = _parse_bool(normalized[field], field=field)
            normalized["gripper_transition_verified_as_grasp_close"] = (
                _parse_optional_bool(
                    normalized["gripper_transition_verified_as_grasp_close"],
                    field="gripper_transition_verified_as_grasp_close",
                )
            )
            normalized["failed_checks"] = split_semicolon(normalized["failed_checks"])
            normalized["observable_modalities"] = split_semicolon(
                normalized["observable_modalities"]
            )
            normalized["parameter_value"] = _parse_optional_float(
                normalized["parameter_value"], field="parameter_value"
            )
            rows.append(PilotRecord.model_validate(normalized))
    return rows


def validate_records_against_plan(
    records: Sequence[PilotRecord],
    plan: Sequence[PilotPlanRow],
) -> None:
    if len(records) != len(plan):
        raise ValueError("pilot records must contain exactly the 12 plan rows")
    plan_by_id = {row.pilot_row_id: row for row in plan}
    seen_episode_ids = set()
    for record in records:
        if record.pilot_row_id not in plan_by_id:
            raise ValueError(f"unknown pilot row {record.pilot_row_id}")
        planned = plan_by_id[record.pilot_row_id]
        for field in (
            "pilot_group_id", "sequence_order", "intensity_rank",
            "intensity_label", "pilot_role", "planned_physical_cause",
            "planned_intervention_type",
        ):
            if getattr(record, field) != getattr(planned, field):
                raise ValueError(
                    f"{record.pilot_row_id}: field {field} differs from plan"
                )
        if record.episode_id in seen_episode_ids:
            raise ValueError("episode_id must be unique across the 12 pilots")
        seen_episode_ids.add(record.episode_id)


def _image_count(path: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(
        1 for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in suffixes
    )


def inspect_raw_episode(
    episode_dir: Path,
    *,
    expected_samples: int,
) -> dict[str, Any]:
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
        return {
            "structure_valid": False,
            "missing": missing,
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = load_state_action_samples(samples_path, verify_tracking_error=True)
    front_count = _image_count(front_dir)
    wrist_count = _image_count(wrist_dir)

    return {
        "structure_valid": (
            len(samples) == expected_samples
            and front_count == expected_samples
            and wrist_count == expected_samples
        ),
        "missing": [],
        "sample_count": len(samples),
        "front_image_count": front_count,
        "wrist_image_count": wrist_count,
        "metadata_schema_version": metadata.get("schema_version"),
        "samples": samples,
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _sustained_first(
    values: Sequence[bool],
    *,
    sustain_frames: int,
) -> int | None:
    if sustain_frames <= 0:
        raise ValueError("sustain_frames must be positive")
    run = 0
    for index, flag in enumerate(values):
        run = run + 1 if flag else 0
        if run >= sustain_frames:
            return index - sustain_frames + 1
    return None


def derive_motion_and_gripper_proxy(
    samples,
    *,
    stable_prefix_frames: int,
    arm_motion_rms_threshold_deg: float,
    gripper_major_transition_min_deg: float,
    gripper_major_transition_fraction_of_range: float,
    sustain_frames: int,
) -> dict[str, Any]:
    if len(samples) <= stable_prefix_frames:
        raise ValueError("episode too short for Day23 proxy")

    arm_joints = tuple(joint for joint in JOINT_ORDER if joint != "gripper")
    baseline = {
        joint: _median(
            [float(getattr(sample.action, joint)) for sample in samples[:stable_prefix_frames]]
        )
        for joint in arm_joints
    }

    arm_flags = []
    for sample in samples:
        diffs = [
            float(getattr(sample.action, joint)) - baseline[joint]
            for joint in arm_joints
        ]
        rms = math.sqrt(sum(value * value for value in diffs) / len(diffs))
        arm_flags.append(rms >= arm_motion_rms_threshold_deg)

    motion_start = _sustained_first(arm_flags, sustain_frames=sustain_frames)

    gripper_values = [float(sample.action.gripper) for sample in samples]
    gripper_baseline = _median(gripper_values[:stable_prefix_frames])
    total_range = max(gripper_values) - min(gripper_values)
    threshold = max(
        gripper_major_transition_min_deg,
        gripper_major_transition_fraction_of_range * total_range,
    )

    gripper_flags = [
        abs(value - gripper_baseline) >= threshold
        for value in gripper_values
    ]
    search_start = motion_start or 0
    suffix_first = _sustained_first(
        gripper_flags[search_start:],
        sustain_frames=sustain_frames,
    )
    gripper_transition = (
        None if suffix_first is None else search_start + suffix_first
    )

    if motion_start is None or gripper_transition is None:
        phase_frames = None
        phase_sec = None
    else:
        phase_frames = gripper_transition - motion_start
        phase_sec = (
            samples[gripper_transition].timestamp_sec
            - samples[motion_start].timestamp_sec
        )

    return {
        "motion_start_frame": motion_start,
        "gripper_major_transition_frame": gripper_transition,
        "gripper_transition_threshold_deg": threshold,
        "gripper_phase_frames_from_motion_start": phase_frames,
        "gripper_phase_sec_from_motion_start": phase_sec,
    }


def build_analysis(
    *,
    records: Sequence[PilotRecord],
    plan: Sequence[PilotPlanRow],
    dataset_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_records_against_plan(records, plan)

    expected_samples = int(config["technical_gate"]["expected_samples"])
    raw_results = {}
    for record in records:
        episode_dir = dataset_root / record.raw_episode_relpath
        inspected = inspect_raw_episode(
            episode_dir,
            expected_samples=expected_samples,
        )
        samples = inspected.pop("samples", None)
        proxy = None
        if samples is not None:
            proxy = derive_motion_and_gripper_proxy(
                samples,
                stable_prefix_frames=int(config["analysis"]["stable_prefix_frames"]),
                arm_motion_rms_threshold_deg=float(
                    config["analysis"]["arm_motion_rms_threshold_deg"]
                ),
                gripper_major_transition_min_deg=float(
                    config["analysis"]["gripper_major_transition_min_deg"]
                ),
                gripper_major_transition_fraction_of_range=float(
                    config["analysis"]["gripper_major_transition_fraction_of_range"]
                ),
                sustain_frames=int(config["analysis"]["sustain_frames"]),
            )
        raw_results[record.pilot_row_id] = {
            "episode_id": record.episode_id,
            "raw_episode_relpath": record.raw_episode_relpath,
            "structure": inspected,
            "proxy": proxy,
        }

    clean_phase_by_group = {}
    for record in records:
        if record.pilot_role != "clean_control":
            continue
        proxy = raw_results[record.pilot_row_id]["proxy"]
        clean_phase_by_group[record.pilot_group_id] = (
            None if proxy is None else proxy["gripper_phase_frames_from_motion_start"]
        )

    for record in records:
        if record.planned_physical_cause != "gripper_close_timing":
            continue
        proxy = raw_results[record.pilot_row_id]["proxy"]
        clean_phase = clean_phase_by_group.get(record.pilot_group_id)
        if proxy is None:
            shift = None
        else:
            phase = proxy["gripper_phase_frames_from_motion_start"]
            shift = None if phase is None or clean_phase is None else phase - clean_phase
        raw_results[record.pilot_row_id]["gripper_shift_frames_vs_pair_clean"] = shift

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "pilot_final_benchmark_eligible": False,
        "dataset_root": str(dataset_root),
        "row_results": raw_results,
    }


def _record_eligible_failure(record: PilotRecord, analysis_row: dict[str, Any]) -> bool:
    return (
        record.recorder_overall_pass
        and not record.failed_checks
        and analysis_row["structure"]["structure_valid"]
        and not record.safety_abort
        and not record.hardware_fault
        and record.intervention_predeclared
        and record.intervention_applied
        and record.single_primary_intervention
        and not record.task_success
        and record.changed_factor_observable
    )


def validate_pilot_and_select(
    *,
    records: Sequence[PilotRecord],
    plan: Sequence[PilotPlanRow],
    analysis: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_records_against_plan(records, plan)
    if analysis.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError("unexpected Day23 analysis schema")

    rows = analysis["row_results"]

    for record in records:
        structure = rows[record.pilot_row_id]["structure"]
        if not record.recorder_overall_pass:
            raise ValueError(
                f"{record.pilot_row_id}: recorder OVERALL EPISODE must PASS"
            )
        if record.failed_checks:
            raise ValueError(
                f"{record.pilot_row_id}: failed_checks must be empty"
            )
        if not structure["structure_valid"]:
            raise ValueError(
                f"{record.pilot_row_id}: raw episode structure invalid"
            )
        if record.safety_abort or record.hardware_fault:
            raise ValueError(
                f"{record.pilot_row_id}: safety/hardware abort invalidates pilot"
            )

    clean_records = [r for r in records if r.pilot_role == "clean_control"]
    if len(clean_records) != 3 or sum(r.task_success for r in clean_records) != 3:
        raise ValueError("Day23 requires 3/3 successful clean controls")

    selected = {}
    cause_counts = {}
    for cause in CAUSES:
        cause_records = [
            r for r in records if r.planned_physical_cause == cause
        ]
        directions = {r.parameter_direction for r in cause_records}
        if len(directions) != 1:
            raise ValueError(f"{cause}: direction must stay fixed across G01/G02/G03")

        eligible = [
            r for r in cause_records
            if _record_eligible_failure(r, rows[r.pilot_row_id])
        ]
        cause_counts[cause] = len(eligible)
        if len(eligible) < int(
            config["parameter_selection"]["per_cause_minimum_eligible_failure_count"]
        ):
            raise ValueError(f"{cause}: requires at least 2/3 eligible induced failures")

        if cause in {"target_offset_or_perception", "trajectory_execution_deviation"}:
            ordered = sorted(cause_records, key=lambda r: r.intensity_rank)
            values = [float(r.parameter_value) for r in ordered]
            if not (values[0] < values[1] < values[2]):
                raise ValueError(f"{cause}: numeric magnitude must increase G01<G02<G03")
            units = {r.parameter_unit for r in cause_records}
            if units != {"mm"}:
                raise ValueError(f"{cause}: parameter unit must be mm")

        if cause == "gripper_close_timing":
            direction = next(iter(directions))
            shifts = []
            for r in sorted(cause_records, key=lambda r: r.intensity_rank):
                if r.gripper_transition_verified_as_grasp_close is not True:
                    raise ValueError(
                        f"{r.pilot_row_id}: gripper transition must be visually verified"
                    )
                shift = rows[r.pilot_row_id].get(
                    "gripper_shift_frames_vs_pair_clean"
                )
                if shift is None or shift == 0:
                    raise ValueError(
                        f"{r.pilot_row_id}: gripper shift could not be derived"
                    )
                if direction == "early" and shift >= 0:
                    raise ValueError(
                        f"{r.pilot_row_id}: early gripper shift must be negative"
                    )
                if direction == "late" and shift <= 0:
                    raise ValueError(
                        f"{r.pilot_row_id}: late gripper shift must be positive"
                    )
                shifts.append(abs(int(shift)))
            if not (shifts[0] < shifts[1] < shifts[2]):
                raise ValueError(
                    "gripper timing magnitude must increase G01<G02<G03"
                )

        eligible_sorted = sorted(
            eligible,
            key=lambda r: (abs(r.intensity_rank - 2), r.intensity_rank),
        )
        chosen = eligible_sorted[0]
        if cause == "gripper_close_timing":
            value = int(
                rows[chosen.pilot_row_id]["gripper_shift_frames_vs_pair_clean"]
            )
            unit = "frames_vs_pair_clean_motion_aligned"
        else:
            value = float(chosen.parameter_value)
            unit = chosen.parameter_unit

        selected[cause] = {
            "selected_pilot_row_id": chosen.pilot_row_id,
            "selected_intensity_rank": chosen.intensity_rank,
            "selected_intensity_label": chosen.intensity_label,
            "direction": chosen.parameter_direction,
            "value": value,
            "unit": unit,
            "eligible_failure_count": len(eligible),
            "selection_rule": (
                "prefer_successful_medium_else_nearest_to_medium_tie_lower"
            ),
        }

    return {
        "schema_version": FREEZE_SCHEMA,
        "freeze_status": FREEZE_STATUS,
        "pilot_final_benchmark_eligible": False,
        "clean_control_success_count": 3,
        "eligible_failure_count_by_cause": cause_counts,
        "selected_parameters": selected,
        "future_split_membership_materialized": False,
        "held_out_data_used": False,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
