from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.state_action_selection import JointVector


class UnifiedEvidenceKind(str, Enum):
    DOCUMENT_PAGE = "document_page"
    ROBOT_SAMPLE = "robot_sample"


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    source_type: SourceType
    manifest_path: str = Field(min_length=1)
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supporting_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_supporting_hashes(self):
        for name, value in self.supporting_sha256.items():
            if not name:
                raise ValueError("supporting hash name must be non-empty")
            if len(value) != 64 or any(
                char not in "0123456789abcdef"
                for char in value
            ):
                raise ValueError(
                    f"supporting_sha256[{name!r}] must be lowercase SHA256"
                )
        return self


class DocumentPagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    char_count: int = Field(ge=0)
    text_excerpt: str = ""
    page_image_path: str | None = None


class RobotCameraAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: Literal["front", "wrist"]
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    image_relpath: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_timestamp_ns: int = Field(gt=0)
    source_age_ms: float = Field(ge=0.0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class RobotStateActionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    observation: JointVector
    action: JointVector
    tracking_error: JointVector


class RobotSamplePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    cameras: list[RobotCameraAsset]
    state_action: RobotStateActionSnapshot

    @model_validator(mode="after")
    def validate_shared_sample(self):
        if [item.camera for item in self.cameras] != ["front", "wrist"]:
            raise ValueError(
                "robot sample cameras must be ordered front, wrist"
            )

        for item in self.cameras:
            if item.frame_index != self.frame_index:
                raise ValueError(
                    "robot camera frame must match shared sample frame"
                )
            if abs(item.timestamp_sec - self.timestamp_sec) > 1e-12:
                raise ValueError(
                    "robot camera timestamp must match shared sample timestamp"
                )

        if self.state_action.frame_index != self.frame_index:
            raise ValueError(
                "state/action frame must match shared sample frame"
            )
        if (
            abs(
                self.state_action.timestamp_sec
                - self.timestamp_sec
            )
            > 1e-12
        ):
            raise ValueError(
                "state/action timestamp must match shared sample timestamp"
            )

        return self


class UnifiedEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    kind: UnifiedEvidenceKind
    refs: list[EvidenceRef]
    provenance: EvidenceProvenance
    payload: DocumentPagePayload | RobotSamplePayload

    @model_validator(mode="after")
    def validate_item_contract(self):
        if not self.refs:
            raise ValueError("unified evidence item requires refs")

        for ref in self.refs:
            if ref.source_id != self.provenance.source_id:
                raise ValueError(
                    "evidence ref source_id must match provenance"
                )
            if ref.source_type != self.provenance.source_type:
                raise ValueError(
                    "evidence ref source_type must match provenance"
                )

        if self.kind == UnifiedEvidenceKind.DOCUMENT_PAGE:
            if not isinstance(self.payload, DocumentPagePayload):
                raise ValueError(
                    "document_page item requires DocumentPagePayload"
                )
            if self.provenance.source_type != SourceType.PDF:
                raise ValueError(
                    "document_page provenance must be PDF"
                )
            if len(self.refs) != 1:
                raise ValueError(
                    "document_page item requires exactly one citation ref"
                )
            ref = self.refs[0]
            if ref.source_type != SourceType.PDF:
                raise ValueError(
                    "document_page ref must be PDF"
                )
            if ref.page_number != self.payload.page_number:
                raise ValueError(
                    "document page ref must match payload page_number"
                )

        elif self.kind == UnifiedEvidenceKind.ROBOT_SAMPLE:
            if not isinstance(self.payload, RobotSamplePayload):
                raise ValueError(
                    "robot_sample item requires RobotSamplePayload"
                )
            if self.provenance.source_type != SourceType.ROBOT_SEQUENCE:
                raise ValueError(
                    "robot_sample provenance must be robot_sequence"
                )
            if self.provenance.source_id != self.payload.episode_id:
                raise ValueError(
                    "robot sample source_id must equal episode_id"
                )
            if len(self.refs) != 2:
                raise ValueError(
                    "robot_sample item requires front and wrist refs"
                )

            cameras = [ref.camera for ref in self.refs]
            if cameras != ["front", "wrist"]:
                raise ValueError(
                    "robot refs must be ordered front, wrist"
                )

            for ref in self.refs:
                if ref.source_type != SourceType.ROBOT_SEQUENCE:
                    raise ValueError(
                        "robot sample refs must be robot_sequence"
                    )
                if ref.frame_index != self.payload.frame_index:
                    raise ValueError(
                        "robot ref frame must match payload frame"
                    )
                if (
                    ref.time_start_sec is None
                    or ref.time_end_sec is None
                ):
                    raise ValueError(
                        "robot frame refs require exact timestamp bounds"
                    )
                if (
                    abs(
                        ref.time_start_sec
                        - self.payload.timestamp_sec
                    )
                    > 1e-12
                    or abs(
                        ref.time_end_sec
                        - self.payload.timestamp_sec
                    )
                    > 1e-12
                ):
                    raise ValueError(
                        "robot ref timestamp must match payload timestamp"
                    )

        return self


class UnifiedEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "evidencemm_unified_evidence_bundle_v1"
    ] = "evidencemm_unified_evidence_bundle_v1"

    bundle_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    items: list[UnifiedEvidenceItem]

    @model_validator(mode="after")
    def validate_bundle(self):
        if not self.items:
            raise ValueError("unified evidence bundle requires items")

        evidence_ids = [item.evidence_id for item in self.items]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError(
                "evidence_id values must be unique within a bundle"
            )

        ref_keys = [
            evidence_ref_key(ref)
            for item in self.items
            for ref in item.refs
        ]
        if len(set(ref_keys)) != len(ref_keys):
            raise ValueError(
                "citation refs must be unique within a bundle"
            )

        return self


class UnifiedGroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    abstain: bool
    citations: list[EvidenceRef]


def evidence_ref_key(ref: EvidenceRef) -> tuple:
    bbox = None
    if ref.bbox is not None:
        bbox = (
            ref.bbox.x1,
            ref.bbox.y1,
            ref.bbox.x2,
            ref.bbox.y2,
        )

    return (
        ref.source_id,
        ref.source_type.value,
        ref.page_number,
        ref.time_start_sec,
        ref.time_end_sec,
        ref.frame_index,
        ref.camera,
        bbox,
    )


def allowed_ref_keys(
    bundle: UnifiedEvidenceBundle,
) -> set[tuple]:
    return {
        evidence_ref_key(ref)
        for item in bundle.items
        for ref in item.refs
    }


def validate_cross_domain_bundle(
    bundle: UnifiedEvidenceBundle,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    kinds = {item.kind for item in bundle.items}

    if UnifiedEvidenceKind.DOCUMENT_PAGE not in kinds:
        errors.append("missing_document_page_evidence")
    if UnifiedEvidenceKind.ROBOT_SAMPLE not in kinds:
        errors.append("missing_robot_sample_evidence")

    return not errors, errors


def validate_unified_citation_policy(
    answer: UnifiedGroundedAnswer,
    bundle: UnifiedEvidenceBundle,
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    allowed = allowed_ref_keys(bundle)
    cited_keys = [
        evidence_ref_key(ref)
        for ref in answer.citations
    ]
    cited = set(cited_keys)

    if len(cited) != len(cited_keys):
        errors.append("duplicate_citations")

    unsupported = cited - allowed
    if unsupported:
        errors.append(
            "citation_outside_supplied_evidence="
            + repr(sorted(unsupported))
        )

    if answer.abstain:
        if answer.citations:
            errors.append(
                "abstention_must_not_emit_citations"
            )
    elif not answer.citations:
        errors.append(
            "answerable_response_requires_citation"
        )

    return not errors, errors
