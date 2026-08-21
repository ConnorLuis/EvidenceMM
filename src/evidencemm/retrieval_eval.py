from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencemm.retrieval_ranking import (
    RankedEvidenceCandidate,
    RetrievalDomain,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalQueryProfile,
)
from evidencemm.unified_evidence import UnifiedEvidenceKind


class Day12RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    document_gold_pages: list[int] = Field(min_length=1)
    expected_robot_joints: list[str] = Field(min_length=1)
    expected_robot_signals: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_gold(self):
        if len(set(self.document_gold_pages)) != len(
            self.document_gold_pages
        ):
            raise ValueError(
                "document_gold_pages must be unique"
            )
        return self


class DocumentRetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_pages: list[int]
    candidate_pool_k: int = Field(ge=1)
    gold_rank: int | None = Field(default=None, ge=1)
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float = Field(ge=0.0, le=1.0)


def evaluate_document_candidates(
    *,
    candidates: list[RankedEvidenceCandidate],
    gold_pages: list[int],
    candidate_pool_k: int,
) -> DocumentRetrievalMetrics:
    if not candidates:
        raise ValueError("document candidates must not be empty")
    if not gold_pages:
        raise ValueError("gold_pages must not be empty")
    if any(
        candidate.domain != RetrievalDomain.DOCUMENT
        for candidate in candidates
    ):
        raise ValueError(
            "document evaluation received non-document candidate"
        )
    if any(
        candidate.item.kind
        != UnifiedEvidenceKind.DOCUMENT_PAGE
        for candidate in candidates
    ):
        raise ValueError(
            "document evaluation requires document_page items"
        )

    gold_set = set(gold_pages)
    gold_rank = None

    for candidate in candidates:
        page_number = candidate.item.payload.page_number
        if page_number in gold_set:
            gold_rank = candidate.rank
            break

    reciprocal_rank = (
        0.0
        if gold_rank is None
        else 1.0 / gold_rank
    )

    return DocumentRetrievalMetrics(
        gold_pages=gold_pages,
        candidate_pool_k=candidate_pool_k,
        gold_rank=gold_rank,
        hit_at_1=(
            gold_rank is not None
            and gold_rank <= 1
        ),
        hit_at_3=(
            gold_rank is not None
            and gold_rank <= 3
        ),
        hit_at_5=(
            gold_rank is not None
            and gold_rank <= 5
        ),
        reciprocal_rank=reciprocal_rank,
    )


def robot_profile_matches(
    *,
    profile: RobotSignalQueryProfile,
    expected_joints: list[str],
    expected_signals: list[str],
) -> bool:
    return (
        list(profile.joints) == expected_joints
        and list(profile.signals) == expected_signals
        and profile.explicit_joint_terms
        and profile.explicit_signal_terms
    )


def validate_robot_candidate_evidence(
    candidates: list[RankedEvidenceCandidate],
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not candidates:
        return False, ["robot candidates must not be empty"]

    for candidate in candidates:
        if candidate.domain != RetrievalDomain.ROBOT:
            errors.append(
                f"rank={candidate.rank}: non-robot domain"
            )
            continue
        if (
            candidate.item.kind
            != UnifiedEvidenceKind.ROBOT_SAMPLE
        ):
            errors.append(
                f"rank={candidate.rank}: non-robot_sample item"
            )
            continue

        payload = candidate.item.payload
        cameras = [
            camera.camera
            for camera in payload.cameras
        ]
        if cameras != ["front", "wrist"]:
            errors.append(
                f"rank={candidate.rank}: cameras={cameras!r}"
            )

        if payload.state_action.frame_index != payload.frame_index:
            errors.append(
                f"rank={candidate.rank}: state/action frame mismatch"
            )

        if abs(
            payload.state_action.timestamp_sec
            - payload.timestamp_sec
        ) > 1e-12:
            errors.append(
                f"rank={candidate.rank}: state/action timestamp mismatch"
            )

    return not errors, errors
