from __future__ import annotations

import csv
import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.schemas import EvidenceRef, SourceType


class AuditCategory(str, Enum):
    CLEAN_REFERENCE_CANDIDATE = "clean_reference_candidate"
    OPERATION_ANOMALY = "operation_anomaly"
    DEMO_QUALITY_ONLY = "demo_quality_only"
    TECHNICAL_EXCLUSION = "technical_exclusion"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class ObservedFailureMode(str, Enum):
    GRASP_DROP = "grasp_drop"
    POST_PLACE_COLLISION = "post_place_collision"
    OBJECT_PUSH_DURING_GRASP = "object_push_during_grasp"
    DROP_ABOVE_TARGET = "drop_above_target"


class CausalDiagnosis(str, Enum):
    TARGET_OFFSET_OR_PERCEPTION = "target_offset_or_perception"
    GRIPPER_CLOSE_TIMING = "gripper_close_timing"
    TRAJECTORY_EXECUTION_DEVIATION = "trajectory_execution_deviation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OTHER = "other"


class ReviewStatus(str, Enum):
    DRAFT = "draft"
    VERIFIED = "verified"


TECHNICAL_MARKERS = (
    "wrist_duplicate_ratio: FAIL",
    "Follower突然断电一般，垂落",
    "Follower突 然断电一般，垂落",
)

DEMO_QUALITY_ONLY_REASONS = {
    "夹起放下过快，导致后续暂停太长",
}

OPERATION_ANOMALY_REASON_MAP: dict[
    str,
    tuple[ObservedFailureMode, ...],
] = {
    "第一次抓取掉落，放入目标区后松开夹爪移出目标区碰到方块": (
        ObservedFailureMode.GRASP_DROP,
        ObservedFailureMode.POST_PLACE_COLLISION,
    ),
    "放入目标区后松开夹爪移出目标区碰到方块": (
        ObservedFailureMode.POST_PLACE_COLLISION,
    ),
    "夹起时推动方块": (
        ObservedFailureMode.OBJECT_PUSH_DURING_GRASP,
    ),
    "方块在目标区上方掉落": (
        ObservedFailureMode.DROP_ABOVE_TARGET,
    ),
    "方块抓取时掉落": (
        ObservedFailureMode.GRASP_DROP,
    ),
}


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"{field} must be boolean-like, got {value!r}"
    )


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "nan":
        return None
    return stripped


class TrainingManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    technical_valid: bool
    task_success: bool
    demo_quality_valid: bool
    valid_for_training: bool
    failure_reason: str | None = None
    notes: str | None = None


class SourceAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "evidencemm_day16_source_audit_v2"
    ] = "evidencemm_day16_source_audit_v2"

    episode_id: str = Field(min_length=1)
    technical_valid: bool
    task_success: bool
    demo_quality_valid: bool
    valid_for_training: bool

    original_failure_reason: str | None = None
    notes: str | None = None

    audit_category: AuditCategory
    operation_anomaly: bool
    observed_failure_modes: list[ObservedFailureMode] = Field(
        default_factory=list
    )

    diagnostic_eligible: bool
    exclusion_reason: str | None = None

    raw_episode_dir: str = Field(min_length=1)
    raw_episode_dir_exists: bool
    metadata_exists: bool
    samples_csv_exists: bool
    front_dir_exists: bool
    wrist_dir_exists: bool

    @model_validator(mode="after")
    def validate_consistency(self):
        if (
            self.audit_category == AuditCategory.OPERATION_ANOMALY
        ) != self.operation_anomaly:
            raise ValueError(
                "operation_anomaly must match audit_category"
            )

        if self.operation_anomaly and not self.observed_failure_modes:
            raise ValueError(
                "operation anomaly requires observed_failure_modes"
            )

        if (
            self.audit_category == AuditCategory.TECHNICAL_EXCLUSION
            and self.diagnostic_eligible
        ):
            raise ValueError(
                "technical exclusion cannot be diagnostic_eligible"
            )

        if (
            self.audit_category == AuditCategory.TECHNICAL_EXCLUSION
            and not self.exclusion_reason
        ):
            raise ValueError(
                "technical exclusion requires exclusion_reason"
            )

        return self


class FailureInterval(BaseModel):
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


class AnomalyEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    observed_failure_mode: ObservedFailureMode
    failure_interval: FailureInterval | None = None
    causal_diagnosis: CausalDiagnosis | None = None
    supporting_robot_refs: list[EvidenceRef] = Field(
        default_factory=list
    )
    counterevidence_robot_refs: list[EvidenceRef] = Field(
        default_factory=list
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    event_status: ReviewStatus = ReviewStatus.DRAFT

    @model_validator(mode="after")
    def validate_event(self):
        for ref in [
            *self.supporting_robot_refs,
            *self.counterevidence_robot_refs,
        ]:
            if ref.source_type != SourceType.ROBOT_SEQUENCE:
                raise ValueError(
                    "event evidence refs must use robot_sequence"
                )

        if self.event_status == ReviewStatus.VERIFIED:
            if self.failure_interval is None:
                raise ValueError(
                    "verified event requires failure_interval"
                )
            if self.causal_diagnosis is None:
                raise ValueError(
                    "verified event requires causal_diagnosis"
                )
            if self.confidence is None:
                raise ValueError(
                    "verified event requires confidence"
                )
            if (
                self.causal_diagnosis
                != CausalDiagnosis.INSUFFICIENT_EVIDENCE
                and not self.supporting_robot_refs
            ):
                raise ValueError(
                    "verified causal event requires supporting_robot_refs"
                )

        return self


class AnomalyReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "evidencemm_day16_anomaly_review_v3"
    ] = "evidencemm_day16_anomaly_review_v3"

    review_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)

    task_success: bool
    operation_anomaly: Literal[True] = True
    original_failure_reason: str = Field(min_length=1)
    events: list[AnomalyEvent] = Field(min_length=1)

    reviewer: str | None = None
    review_notes: str | None = None

    diagnostic_manifest_path: str = Field(min_length=1)
    diagnostic_frames_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review(self):
        event_ids = [
            event.event_id
            for event in self.events
        ]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(
                "event_id values must be unique within an episode"
            )

        for event in self.events:
            expected_prefix = (
                f"{self.episode_id}_event_"
            )
            if not event.event_id.startswith(
                expected_prefix
            ):
                raise ValueError(
                    "event_id must be namespaced by episode_id"
                )

            for ref in [
                *event.supporting_robot_refs,
                *event.counterevidence_robot_refs,
            ]:
                if ref.source_id != self.episode_id:
                    raise ValueError(
                        "event robot ref source_id must match episode_id"
                    )

        if any(
            event.event_status == ReviewStatus.VERIFIED
            for event in self.events
        ) and not self.reviewer:
            raise ValueError(
                "any verified event requires reviewer"
            )

        return self

    @property
    def observed_failure_modes(
        self,
    ) -> list[ObservedFailureMode]:
        return [
            event.observed_failure_mode
            for event in self.events
        ]

    @property
    def all_events_draft(self) -> bool:
        return all(
            event.event_status == ReviewStatus.DRAFT
            for event in self.events
        )

    @property
    def all_causal_diagnoses_unset(self) -> bool:
        return all(
            event.causal_diagnosis is None
            for event in self.events
        )

    @property
    def manual_review_complete(self) -> bool:
        return all(
            event.event_status == ReviewStatus.VERIFIED
            for event in self.events
        )


def load_training_manifest(
    path: str | Path,
    *,
    encoding: str = "gb18030",
) -> list[TrainingManifestRow]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    with file_path.open(
        encoding=encoding,
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        required = {
            "episode_id",
            "technical_valid",
            "task_success",
            "demo_quality_valid",
            "valid_for_training",
            "failure_reason",
            "notes",
        }
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                "training_manifest.csv missing columns: "
                + repr(sorted(missing))
            )

        rows = [
            TrainingManifestRow(
                episode_id=row["episode_id"].strip(),
                technical_valid=_parse_bool(
                    row["technical_valid"],
                    field="technical_valid",
                ),
                task_success=_parse_bool(
                    row["task_success"],
                    field="task_success",
                ),
                demo_quality_valid=_parse_bool(
                    row["demo_quality_valid"],
                    field="demo_quality_valid",
                ),
                valid_for_training=_parse_bool(
                    row["valid_for_training"],
                    field="valid_for_training",
                ),
                failure_reason=_optional_text(
                    row.get("failure_reason")
                ),
                notes=_optional_text(
                    row.get("notes")
                ),
            )
            for row in reader
        ]

    if not rows:
        raise ValueError(
            "training_manifest.csv contains no rows"
        )

    ids = [row.episode_id for row in rows]
    duplicates = sorted(
        episode_id
        for episode_id, count
        in Counter(ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            "duplicate episode_id values: "
            + repr(duplicates)
        )

    return rows


def classify_training_row(
    row: TrainingManifestRow,
) -> tuple[
    AuditCategory,
    list[ObservedFailureMode],
    bool,
    str | None,
]:
    reason = row.failure_reason

    if (
        not row.technical_valid
        or (
            reason is not None
            and any(
                marker in reason
                for marker in TECHNICAL_MARKERS
            )
        )
    ):
        exclusion_reason = (
            reason
            or "technical_valid=False"
        )
        return (
            AuditCategory.TECHNICAL_EXCLUSION,
            [],
            False,
            exclusion_reason,
        )

    if reason in OPERATION_ANOMALY_REASON_MAP:
        return (
            AuditCategory.OPERATION_ANOMALY,
            list(
                OPERATION_ANOMALY_REASON_MAP[reason]
            ),
            True,
            None,
        )

    if (
        reason in DEMO_QUALITY_ONLY_REASONS
        or (
            not row.demo_quality_valid
            and reason is None
        )
    ):
        return (
            AuditCategory.DEMO_QUALITY_ONLY,
            [],
            True,
            None,
        )

    if reason is None:
        return (
            AuditCategory.CLEAN_REFERENCE_CANDIDATE,
            [],
            True,
            None,
        )

    return (
        AuditCategory.MANUAL_REVIEW_REQUIRED,
        [],
        True,
        None,
    )


def audit_training_row(
    row: TrainingManifestRow,
    *,
    dataset_root: str | Path,
) -> SourceAuditRecord:
    root = Path(dataset_root)
    episode_dir = root / row.episode_id

    (
        category,
        modes,
        diagnostic_eligible,
        exclusion_reason,
    ) = classify_training_row(row)

    return SourceAuditRecord(
        episode_id=row.episode_id,
        technical_valid=row.technical_valid,
        task_success=row.task_success,
        demo_quality_valid=row.demo_quality_valid,
        valid_for_training=row.valid_for_training,
        original_failure_reason=row.failure_reason,
        notes=row.notes,
        audit_category=category,
        operation_anomaly=(
            category == AuditCategory.OPERATION_ANOMALY
        ),
        observed_failure_modes=modes,
        diagnostic_eligible=diagnostic_eligible,
        exclusion_reason=exclusion_reason,
        raw_episode_dir=str(episode_dir),
        raw_episode_dir_exists=episode_dir.is_dir(),
        metadata_exists=(
            episode_dir / "metadata.json"
        ).is_file(),
        samples_csv_exists=(
            episode_dir / "samples.csv"
        ).is_file(),
        front_dir_exists=(
            episode_dir / "front"
        ).is_dir(),
        wrist_dir_exists=(
            episode_dir / "wrist"
        ).is_dir(),
    )


def source_presence_complete(
    record: SourceAuditRecord,
) -> bool:
    return (
        record.raw_episode_dir_exists
        and record.metadata_exists
        and record.samples_csv_exists
        and record.front_dir_exists
        and record.wrist_dir_exists
    )


def required_anomaly_source_presence_complete(
    records: list[SourceAuditRecord],
) -> bool:
    anomalies = [
        record
        for record in records
        if (
            record.audit_category
            == AuditCategory.OPERATION_ANOMALY
        )
    ]
    return bool(anomalies) and all(
        source_presence_complete(record)
        for record in anomalies
    )


def build_anomaly_review_case(
    audit: SourceAuditRecord,
    *,
    diagnostic_manifest_root: str,
    diagnostic_processed_root: str,
) -> AnomalyReviewCase:
    if (
        audit.audit_category
        != AuditCategory.OPERATION_ANOMALY
    ):
        raise ValueError(
            "review case requires operation_anomaly audit record"
        )
    if audit.original_failure_reason is None:
        raise ValueError(
            "operation anomaly requires original failure reason"
        )

    events = [
        AnomalyEvent(
            event_id=(
                f"{audit.episode_id}_event_{index:02d}"
            ),
            observed_failure_mode=mode,
        )
        for index, mode in enumerate(
            audit.observed_failure_modes,
            start=1,
        )
    ]

    return AnomalyReviewCase(
        review_id=f"day16_review_{audit.episode_id}",
        episode_id=audit.episode_id,
        task_success=audit.task_success,
        original_failure_reason=(
            audit.original_failure_reason
        ),
        events=events,
        reviewer=None,
        review_notes=None,
        diagnostic_manifest_path=(
            f"{diagnostic_manifest_root}/"
            f"{audit.episode_id}.json"
        ),
        diagnostic_frames_path=(
            f"{diagnostic_processed_root}/"
            f"{audit.episode_id}/frames.jsonl"
        ),
    )


def save_jsonl(
    items: list[BaseModel],
    path: str | Path,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    file_path.write_text(
        "\n".join(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
            )
            for item in items
        )
        + ("\n" if items else ""),
        encoding="utf-8",
        newline="\n",
    )


def load_source_audit(
    path: str | Path,
) -> list[SourceAuditRecord]:
    return _load_jsonl(
        path,
        SourceAuditRecord,
    )


def load_anomaly_review_cases(
    path: str | Path,
) -> list[AnomalyReviewCase]:
    return _load_jsonl(
        path,
        AnomalyReviewCase,
    )


def _load_jsonl(
    path: str | Path,
    model,
):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return [
        model.model_validate(
            json.loads(line)
        )
        for line in file_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def summarize_source_audit(
    records: list[SourceAuditRecord],
) -> dict:
    return {
        "total_rows": len(records),
        "categories": dict(
            sorted(
                Counter(
                    item.audit_category.value
                    for item in records
                ).items()
            )
        ),
        "task_success": dict(
            sorted(
                Counter(
                    str(item.task_success).lower()
                    for item in records
                ).items()
            )
        ),
        "operation_anomaly": dict(
            sorted(
                Counter(
                    str(item.operation_anomaly).lower()
                    for item in records
                ).items()
            )
        ),
        "operation_anomaly_task_success": dict(
            sorted(
                Counter(
                    str(item.task_success).lower()
                    for item in records
                    if item.operation_anomaly
                ).items()
            )
        ),
        "failure_reasons": dict(
            sorted(
                Counter(
                    item.original_failure_reason
                    for item in records
                    if item.original_failure_reason
                    is not None
                ).items()
            )
        ),
        "observed_failure_modes": dict(
            sorted(
                Counter(
                    mode.value
                    for item in records
                    for mode in item.observed_failure_modes
                ).items()
            )
        ),
        "source_presence": {
            "episode_dir": sum(
                item.raw_episode_dir_exists
                for item in records
            ),
            "metadata_json": sum(
                item.metadata_exists
                for item in records
            ),
            "samples_csv": sum(
                item.samples_csv_exists
                for item in records
            ),
            "front_dir": sum(
                item.front_dir_exists
                for item in records
            ),
            "wrist_dir": sum(
                item.wrist_dir_exists
                for item in records
            ),
        },
        "operation_anomaly_episode_ids": [
            item.episode_id
            for item in records
            if item.operation_anomaly
        ],
        "technical_exclusion_episode_ids": [
            item.episode_id
            for item in records
            if (
                item.audit_category
                == AuditCategory.TECHNICAL_EXCLUSION
            )
        ],
    }
