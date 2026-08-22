from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


CONFIG_SCHEMA = "evidencemm_day22_root_cause_benchmark_v2_config_v1"
PROTOCOL_SCHEMA = "evidencemm_day22_root_cause_benchmark_v2_protocol_v1"
PROTOCOL_STATUS = "root_cause_benchmark_v2_protocol_frozen_pre_collection"
PLAN_SCHEMA = "evidencemm_day22_root_cause_collection_plan_v1"

PAIR_GROUP_COUNT = 15
SLOTS_PER_GROUP = 6
TARGET_EPISODE_COUNT = 90

CAUSES = (
    "target_offset_or_perception",
    "gripper_close_timing",
    "trajectory_execution_deviation",
)
CAUSE_TO_INTERVENTION = {
    "target_offset_or_perception": "object_target_pose_offset",
    "gripper_close_timing": "manual_gripper_close_timing_shift",
    "trajectory_execution_deviation": "manual_bounded_trajectory_deviation",
}
REPEAT_CAUSE_ROTATION = CAUSES

PLAN_COLUMNS = (
    "schema_version",
    "plan_row_id",
    "pair_group_id",
    "slot_index",
    "slot_role",
    "planned_physical_cause",
    "planned_intervention_type",
    "repeat_slot",
    "benchmark_target",
    "episode_id",
    "collection_status",
    "intervention_applied",
    "intervention_parameter_json",
    "operator_notes",
)


class PhysicalCauseGT(str, Enum):
    TARGET_OFFSET_OR_PERCEPTION = "target_offset_or_perception"
    GRIPPER_CLOSE_TIMING = "gripper_close_timing"
    TRAJECTORY_EXECUTION_DEVIATION = "trajectory_execution_deviation"
    UNKNOWN = "unknown"
    NONE_CLEAN = "none_clean"


class EvidenceAnswerabilityGT(str, Enum):
    ANSWERABLE = "answerable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE_CLEAN = "not_applicable_clean"


class DiagnosticDecisionGT(str, Enum):
    TARGET_OFFSET_OR_PERCEPTION = "target_offset_or_perception"
    GRIPPER_CLOSE_TIMING = "gripper_close_timing"
    TRAJECTORY_EXECUTION_DEVIATION = "trajectory_execution_deviation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CLEAN_SUCCESS = "clean_success"


class CollectionPlanRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "evidencemm_day22_root_cause_collection_plan_v1"
    ] = PLAN_SCHEMA
    plan_row_id: str = Field(pattern=r"^rcv2_g\d{2}_s\d{2}$")
    pair_group_id: str = Field(pattern=r"^rcv2_g\d{2}$")
    slot_index: int = Field(ge=1, le=6)
    slot_role: Literal[
        "clean_control",
        "controlled_cause",
        "insufficient_evidence_candidate",
    ]
    planned_physical_cause: Literal[
        "target_offset_or_perception",
        "gripper_close_timing",
        "trajectory_execution_deviation",
        "unknown",
        "none_clean",
    ]
    planned_intervention_type: Literal[
        "none",
        "object_target_pose_offset",
        "manual_gripper_close_timing_shift",
        "manual_bounded_trajectory_deviation",
        "ambiguity_protocol",
    ]
    repeat_slot: bool
    benchmark_target: Literal["final_v2"]
    episode_id: str = ""
    collection_status: Literal["pending"] = "pending"
    intervention_applied: str = ""
    intervention_parameter_json: str = ""
    operator_notes: str = ""

    @model_validator(mode="after")
    def validate_role(self):
        expected_id = (
            f"{self.pair_group_id}_s{self.slot_index:02d}"
        )
        if self.plan_row_id != expected_id:
            raise ValueError(
                "plan_row_id must equal pair_group_id + slot index"
            )

        if self.slot_role == "clean_control":
            if self.planned_physical_cause != "none_clean":
                raise ValueError(
                    "clean control must use none_clean"
                )
            if self.planned_intervention_type != "none":
                raise ValueError(
                    "clean control must not declare intervention"
                )
            if self.repeat_slot:
                raise ValueError(
                    "clean control cannot be repeat slot"
                )

        elif self.slot_role == "controlled_cause":
            if self.planned_physical_cause not in CAUSES:
                raise ValueError(
                    "controlled cause requires one physical cause"
                )
            expected = CAUSE_TO_INTERVENTION[
                self.planned_physical_cause
            ]
            if self.planned_intervention_type != expected:
                raise ValueError(
                    "controlled cause intervention type mismatch"
                )

        elif self.slot_role == "insufficient_evidence_candidate":
            if self.planned_physical_cause != "unknown":
                raise ValueError(
                    "insufficient candidate physical cause is not GT yet"
                )
            if (
                self.planned_intervention_type
                != "ambiguity_protocol"
            ):
                raise ValueError(
                    "insufficient candidate must use ambiguity protocol"
                )
            if self.repeat_slot:
                raise ValueError(
                    "insufficient candidate cannot be repeat slot"
                )

        return self


class FailureIntervalV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.end_frame < self.start_frame:
            raise ValueError(
                "end_frame must be >= start_frame"
            )
        if self.end_sec < self.start_sec:
            raise ValueError(
                "end_sec must be >= start_sec"
            )
        return self


class RootCauseReviewRecordV2(BaseModel):
    """Frozen Day22 semantic contract for future Day29 human review.

    Evidence references are kept as JSON-compatible dictionaries here because
    the protocol freezes semantics before the final review tool is implemented.
    Day29 must resolve them to the canonical EvidenceRef contract.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "evidencemm_root_cause_review_record_v2"
    ] = "evidencemm_root_cause_review_record_v2"

    episode_id: str = Field(min_length=1)
    pair_group_id: str = Field(pattern=r"^rcv2_g\d{2}$")

    technical_valid: bool
    experimental_valid: bool
    task_success: bool
    intervention_verified: bool | None

    observed_failure_description: str | None = None
    physical_cause_gt: PhysicalCauseGT
    evidence_answerability_gt: EvidenceAnswerabilityGT
    diagnostic_decision_gt: DiagnosticDecisionGT

    failure_interval: FailureIntervalV2 | None = None

    supporting_robot_refs: list[dict[str, Any]] = Field(
        default_factory=list
    )
    counterevidence_robot_refs: list[dict[str, Any]] = Field(
        default_factory=list
    )
    supporting_manual_refs: list[dict[str, Any]] = Field(
        default_factory=list
    )
    counterevidence_manual_refs: list[dict[str, Any]] = Field(
        default_factory=list
    )

    confidence: float = Field(ge=0.0, le=1.0)
    review_notes: str = Field(min_length=1)
    review_status: Literal["verified"] = "verified"

    @model_validator(mode="after")
    def validate_semantics(self):
        if not self.technical_valid:
            raise ValueError(
                "verified benchmark record cannot be technical exclusion"
            )
        if not self.experimental_valid:
            raise ValueError(
                "verified benchmark record cannot be experimental exclusion"
            )

        if self.diagnostic_decision_gt == DiagnosticDecisionGT.CLEAN_SUCCESS:
            if not self.task_success:
                raise ValueError(
                    "clean_success requires task_success"
                )
            if self.physical_cause_gt != PhysicalCauseGT.NONE_CLEAN:
                raise ValueError(
                    "clean_success requires physical_cause_gt=none_clean"
                )
            if (
                self.evidence_answerability_gt
                != EvidenceAnswerabilityGT.NOT_APPLICABLE_CLEAN
            ):
                raise ValueError(
                    "clean_success requires not_applicable_clean"
                )
            if self.failure_interval is not None:
                raise ValueError(
                    "clean_success must not invent failure interval"
                )
            if self.intervention_verified not in {False, None}:
                raise ValueError(
                    "clean_success cannot verify a causal intervention"
                )
            return self

        if self.task_success:
            raise ValueError(
                "failure diagnosis record requires task_success=false"
            )
        if self.failure_interval is None:
            raise ValueError(
                "failed benchmark record requires failure_interval"
            )

        if (
            self.diagnostic_decision_gt
            == DiagnosticDecisionGT.INSUFFICIENT_EVIDENCE
        ):
            if (
                self.evidence_answerability_gt
                != EvidenceAnswerabilityGT.INSUFFICIENT_EVIDENCE
            ):
                raise ValueError(
                    "insufficient decision requires insufficient_evidence"
                )
            if self.physical_cause_gt == PhysicalCauseGT.NONE_CLEAN:
                raise ValueError(
                    "failed insufficient case cannot use none_clean"
                )
            return self

        expected_physical = PhysicalCauseGT(
            self.diagnostic_decision_gt.value
        )
        if self.physical_cause_gt != expected_physical:
            raise ValueError(
                "answerable diagnostic decision must equal physical_cause_gt"
            )
        if (
            self.evidence_answerability_gt
            != EvidenceAnswerabilityGT.ANSWERABLE
        ):
            raise ValueError(
                "physical cause decision requires answerable evidence"
            )
        if self.intervention_verified is not True:
            raise ValueError(
                "answerable controlled cause requires verified intervention"
            )
        if not self.supporting_robot_refs:
            raise ValueError(
                "answerable physical cause requires supporting robot refs"
            )
        if not self.supporting_manual_refs:
            raise ValueError(
                "answerable physical cause requires supporting manual refs"
            )
        return self


def repeat_cause_for_group(group_index: int) -> str:
    if group_index < 1 or group_index > PAIR_GROUP_COUNT:
        raise ValueError(
            f"group_index must be 1..{PAIR_GROUP_COUNT}"
        )
    return REPEAT_CAUSE_ROTATION[
        (group_index - 1) % len(REPEAT_CAUSE_ROTATION)
    ]


def build_collection_plan() -> list[CollectionPlanRow]:
    rows: list[CollectionPlanRow] = []

    for group_index in range(
        1,
        PAIR_GROUP_COUNT + 1,
    ):
        group_id = f"rcv2_g{group_index:02d}"
        repeat_cause = repeat_cause_for_group(
            group_index
        )

        specs = [
            (
                "clean_control",
                "none_clean",
                "none",
                False,
            ),
            (
                "controlled_cause",
                "target_offset_or_perception",
                "object_target_pose_offset",
                False,
            ),
            (
                "controlled_cause",
                "gripper_close_timing",
                "manual_gripper_close_timing_shift",
                False,
            ),
            (
                "controlled_cause",
                "trajectory_execution_deviation",
                "manual_bounded_trajectory_deviation",
                False,
            ),
            (
                "controlled_cause",
                repeat_cause,
                CAUSE_TO_INTERVENTION[
                    repeat_cause
                ],
                True,
            ),
            (
                "insufficient_evidence_candidate",
                "unknown",
                "ambiguity_protocol",
                False,
            ),
        ]

        for slot_index, (
            role,
            cause,
            intervention,
            repeat_slot,
        ) in enumerate(
            specs,
            start=1,
        ):
            rows.append(
                CollectionPlanRow(
                    plan_row_id=(
                        f"{group_id}_s{slot_index:02d}"
                    ),
                    pair_group_id=group_id,
                    slot_index=slot_index,
                    slot_role=role,
                    planned_physical_cause=cause,
                    planned_intervention_type=intervention,
                    repeat_slot=repeat_slot,
                    benchmark_target="final_v2",
                )
            )

    validate_collection_plan(rows)
    return rows


def validate_collection_plan(
    rows: Sequence[CollectionPlanRow],
) -> None:
    if len(rows) != TARGET_EPISODE_COUNT:
        raise ValueError(
            f"expected {TARGET_EPISODE_COUNT} plan rows, got {len(rows)}"
        )

    row_ids = [
        row.plan_row_id
        for row in rows
    ]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(
            "collection plan row IDs must be unique"
        )

    by_group: dict[
        str,
        list[CollectionPlanRow],
    ] = defaultdict(list)
    for row in rows:
        by_group[row.pair_group_id].append(
            row
        )

    if len(by_group) != PAIR_GROUP_COUNT:
        raise ValueError(
            f"expected {PAIR_GROUP_COUNT} pair groups"
        )

    cause_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    for group_id, group_rows in sorted(
        by_group.items()
    ):
        if len(group_rows) != SLOTS_PER_GROUP:
            raise ValueError(
                f"{group_id}: expected {SLOTS_PER_GROUP} slots"
            )
        indices = sorted(
            row.slot_index
            for row in group_rows
        )
        if indices != list(
            range(
                1,
                SLOTS_PER_GROUP + 1,
            )
        ):
            raise ValueError(
                f"{group_id}: slot indices must be 1..6"
            )

        group_roles = Counter(
            row.slot_role
            for row in group_rows
        )
        if group_roles != Counter(
            {
                "clean_control": 1,
                "controlled_cause": 4,
                "insufficient_evidence_candidate": 1,
            }
        ):
            raise ValueError(
                f"{group_id}: invalid role composition"
            )

        core_causes = Counter(
            row.planned_physical_cause
            for row in group_rows
            if (
                row.slot_role == "controlled_cause"
                and not row.repeat_slot
            )
        )
        if core_causes != Counter(
            {cause: 1 for cause in CAUSES}
        ):
            raise ValueError(
                f"{group_id}: each core cause must appear once"
            )

        repeat_rows = [
            row
            for row in group_rows
            if row.repeat_slot
        ]
        if len(repeat_rows) != 1:
            raise ValueError(
                f"{group_id}: expected exactly one repeat slot"
            )

        for row in group_rows:
            role_counts[row.slot_role] += 1
            if row.slot_role == "controlled_cause":
                cause_counts[
                    row.planned_physical_cause
                ] += 1

    if cause_counts != Counter(
        {
            "target_offset_or_perception": 20,
            "gripper_close_timing": 20,
            "trajectory_execution_deviation": 20,
        }
    ):
        raise ValueError(
            f"unexpected controlled-cause counts: {cause_counts}"
        )

    if role_counts != Counter(
        {
            "controlled_cause": 60,
            "clean_control": 15,
            "insufficient_evidence_candidate": 15,
        }
    ):
        raise ValueError(
            f"unexpected role counts: {role_counts}"
        )


def collection_plan_csv_bytes(
    rows: Sequence[CollectionPlanRow],
) -> bytes:
    validate_collection_plan(
        rows
    )
    stream = io.StringIO(
        newline=""
    )
    writer = csv.DictWriter(
        stream,
        fieldnames=list(
            PLAN_COLUMNS
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        payload = row.model_dump(
            mode="json"
        )
        writer.writerow(
            {
                column: (
                    json.dumps(
                        payload[column],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if column == "intervention_parameter_json"
                    and isinstance(
                        payload[column],
                        (dict, list),
                    )
                    else payload[column]
                )
                for column in PLAN_COLUMNS
            }
        )
    return stream.getvalue().encode(
        "utf-8"
    )


def load_collection_plan_csv(
    content: str,
) -> list[CollectionPlanRow]:
    reader = csv.DictReader(
        io.StringIO(content)
    )
    if tuple(
        reader.fieldnames or ()
    ) != PLAN_COLUMNS:
        raise ValueError(
            "collection plan CSV columns differ from frozen contract"
        )

    rows = []
    for raw in reader:
        normalized = dict(
            raw
        )
        normalized["slot_index"] = int(
            normalized["slot_index"]
        )
        normalized["repeat_slot"] = (
            normalized[
                "repeat_slot"
            ].strip().lower()
            == "true"
        )
        rows.append(
            CollectionPlanRow.model_validate(
                normalized
            )
        )
    validate_collection_plan(
        rows
    )
    return rows


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def future_split_rank(
    pair_group_id: str,
    *,
    seed: str,
) -> str:
    return hashlib.sha256(
        f"{seed}|{pair_group_id}".encode(
            "utf-8"
        )
    ).hexdigest()


def materialize_future_group_split(
    pair_group_ids: Sequence[str],
    *,
    seed: str,
    held_out_group_count: int,
) -> tuple[list[str], list[str]]:
    unique = sorted(
        set(pair_group_ids)
    )
    if len(unique) != len(
        pair_group_ids
    ):
        raise ValueError(
            "pair_group_ids must be unique"
        )
    if held_out_group_count <= 0:
        raise ValueError(
            "held_out_group_count must be positive"
        )
    if held_out_group_count >= len(
        unique
    ):
        raise ValueError(
            "held_out_group_count must be smaller than group count"
        )

    ranked = sorted(
        unique,
        key=lambda group_id: (
            future_split_rank(
                group_id,
                seed=seed,
            ),
            group_id,
        ),
    )
    held_out = sorted(
        ranked[
            :held_out_group_count
        ]
    )
    development = sorted(
        ranked[
            held_out_group_count:
        ]
    )
    return development, held_out


def collection_plan_summary(
    rows: Sequence[CollectionPlanRow],
) -> dict[str, Any]:
    validate_collection_plan(
        rows
    )
    role_counts = Counter(
        row.slot_role
        for row in rows
    )
    cause_counts = Counter(
        row.planned_physical_cause
        for row in rows
        if row.slot_role
        == "controlled_cause"
    )
    return {
        "pair_group_count": (
            PAIR_GROUP_COUNT
        ),
        "slots_per_pair_group": (
            SLOTS_PER_GROUP
        ),
        "target_episode_count": (
            TARGET_EPISODE_COUNT
        ),
        "role_counts": dict(
            sorted(
                role_counts.items()
            )
        ),
        "controlled_cause_counts": {
            cause: int(
                cause_counts[cause]
            )
            for cause in CAUSES
        },
    }


def build_protocol_artifact(
    *,
    config: dict[str, Any],
    collection_plan_sha256: str,
    frozen_blob_sha1: dict[str, str],
) -> dict[str, Any]:
    if config.get(
        "schema_version"
    ) != CONFIG_SCHEMA:
        raise ValueError(
            "unexpected Day22 config schema_version"
        )

    final = config[
        "final_collection"
    ]
    expected_counts = {
        "pair_group_count": (
            PAIR_GROUP_COUNT
        ),
        "slots_per_pair_group": (
            SLOTS_PER_GROUP
        ),
        "target_eligible_episode_count": (
            TARGET_EPISODE_COUNT
        ),
        "clean_control_count": 15,
        "insufficient_evidence_candidate_count": 15,
        "controlled_cause_count": 60,
        "target_offset_or_perception_count": 20,
        "gripper_close_timing_count": 20,
        "trajectory_execution_deviation_count": 20,
    }
    for key, expected in expected_counts.items():
        if int(
            final[key]
        ) != expected:
            raise ValueError(
                f"Day22 final collection count mismatch for {key}"
            )

    future_split = config[
        "future_split"
    ]
    if (
        future_split[
            "materialize_membership_on_day22"
        ]
        is not False
    ):
        raise ValueError(
            "Day22 must not materialize future split membership"
        )
    if (
        future_split[
            "pair_group_cross_split_allowed"
        ]
        is not False
    ):
        raise ValueError(
            "pair groups must remain split-atomic"
        )

    leakage = config[
        "anti_label_leakage"
    ]
    if (
        leakage[
            "source_manifest_must_not_embed_admin_labels"
        ]
        is not True
    ):
        raise ValueError(
            "source manifests must exclude admin labels"
        )

    return {
        "schema_version": (
            PROTOCOL_SCHEMA
        ),
        "protocol_status": (
            PROTOCOL_STATUS
        ),
        "scope": (
            "prospective_controlled_intervention_root_cause_"
            "benchmark_protocol_pre_collection"
        ),
        "provenance": {
            "frozen_after_day21_commit": (
                config["provenance"][
                    "frozen_after_day21_commit"
                ]
            ),
            "protocol_frozen_date": (
                config["provenance"][
                    "protocol_frozen_date"
                ]
            ),
            "task_definition_blob_sha1": (
                frozen_blob_sha1[
                    "task_definition"
                ]
            ),
            "day21_doc_blob_sha1": (
                frozen_blob_sha1[
                    "day21_doc"
                ]
            ),
            "day21_artifact_blob_sha1": (
                frozen_blob_sha1[
                    "day21_artifact"
                ]
            ),
            "collection_plan_sha256": (
                collection_plan_sha256
            ),
        },
        "ownership": config[
            "ownership"
        ],
        "taxonomy": config[
            "taxonomy"
        ],
        "pilot": config[
            "pilot"
        ],
        "final_collection": config[
            "final_collection"
        ],
        "collection_plan_contract": {
            "schema_version": (
                PLAN_SCHEMA
            ),
            "pair_group_count": (
                PAIR_GROUP_COUNT
            ),
            "slots_per_pair_group": (
                SLOTS_PER_GROUP
            ),
            "slot_layout": [
                "clean_control",
                "target_offset_or_perception",
                "gripper_close_timing",
                "trajectory_execution_deviation",
                "rotating_repeat_controlled_cause",
                "insufficient_evidence_candidate",
            ],
            "repeat_cause_rotation": list(
                REPEAT_CAUSE_ROTATION
            ),
            "future_split_membership_materialized": False,
        },
        "interventions": config[
            "interventions"
        ],
        "single_primary_intervention": (
            config[
                "single_primary_intervention"
            ]
        ),
        "paired_control": config[
            "paired_control"
        ],
        "insufficient_evidence": config[
            "insufficient_evidence"
        ],
        "technical_exclusion": config[
            "technical_exclusion"
        ],
        "experimental_exclusion": config[
            "experimental_exclusion"
        ],
        "human_review": config[
            "human_review"
        ],
        "review_record_contract": {
            "schema_version": (
                "evidencemm_root_cause_review_record_v2"
            ),
            "physical_cause_values": [
                item.value
                for item in PhysicalCauseGT
            ],
            "evidence_answerability_values": [
                item.value
                for item in EvidenceAnswerabilityGT
            ],
            "diagnostic_decision_values": [
                item.value
                for item in DiagnosticDecisionGT
            ],
            "decision_rule": (
                "if clean: clean_success; else if evidence is not "
                "sufficient: insufficient_evidence; else diagnostic "
                "decision must equal verified physical_cause_gt"
            ),
        },
        "manual_causal_ground_truth": (
            config[
                "manual_causal_ground_truth"
            ]
        ),
        "anti_label_leakage": (
            config[
                "anti_label_leakage"
            ]
        ),
        "future_split": config[
            "future_split"
        ],
        "final_metrics": config[
            "final_metrics"
        ],
        "safety": config[
            "safety"
        ],
        "roadmap": config[
            "roadmap"
        ],
        "acceptance": {
            "day22_freezes_numeric_intervention_values": False,
            "day23_pilot_must_freeze_numeric_intervention_values": True,
            "pilot_episodes_are_final_benchmark_eligible": False,
            "final_benchmark_target_eligible_episode_count": 90,
            "future_split_membership_is_not_materialized": True,
            "physical_root_cause_model_is_not_trained_on_day22": True,
        },
    }


def validate_protocol_artifact(
    artifact: dict[str, Any],
    *,
    expected_plan_sha256: str,
) -> None:
    if artifact.get(
        "schema_version"
    ) != PROTOCOL_SCHEMA:
        raise ValueError(
            "unexpected Day22 protocol schema_version"
        )
    if artifact.get(
        "protocol_status"
    ) != PROTOCOL_STATUS:
        raise ValueError(
            "unexpected Day22 protocol_status"
        )

    provenance = artifact[
        "provenance"
    ]
    if provenance[
        "collection_plan_sha256"
    ] != expected_plan_sha256:
        raise ValueError(
            "collection plan SHA256 mismatch"
        )

    acceptance = artifact[
        "acceptance"
    ]
    expected_flags = {
        "day22_freezes_numeric_intervention_values": False,
        "day23_pilot_must_freeze_numeric_intervention_values": True,
        "pilot_episodes_are_final_benchmark_eligible": False,
        "future_split_membership_is_not_materialized": True,
        "physical_root_cause_model_is_not_trained_on_day22": True,
    }
    for key, expected in expected_flags.items():
        if acceptance.get(key) is not expected:
            raise ValueError(
                f"Day22 acceptance flag mismatch: {key}"
            )

    if int(
        acceptance[
            "final_benchmark_target_eligible_episode_count"
        ]
    ) != TARGET_EPISODE_COUNT:
        raise ValueError(
            "Day22 target eligible episode count mismatch"
        )

    split = artifact[
        "future_split"
    ]
    if (
        split[
            "materialize_membership_on_day22"
        ]
        is not False
    ):
        raise ValueError(
            "future split unexpectedly materialized"
        )
    if (
        split[
            "held_out_model_selection_allowed"
        ]
        is not False
    ):
        raise ValueError(
            "held-out model selection must be forbidden"
        )

    safety = artifact[
        "safety"
    ]
    for key in (
        "preserve_existing_robot_safety_guards",
        "no_safety_limit_bypass",
        "no_deliberate_power_loss",
        "no_hard_collision_as_intervention",
        "intervention_must_remain_inside_existing_safe_workspace",
        "unexpected_contact_or_hardware_fault_requires_abort_and_exclusion",
    ):
        if safety.get(key) is not True:
            raise ValueError(
                f"required safety constraint missing: {key}"
            )


def canonical_json_bytes(
    value: Any,
) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode(
        "utf-8"
    )
