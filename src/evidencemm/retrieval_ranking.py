from __future__ import annotations

import math
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from evidencemm.retrieval import normalize_query
from evidencemm.unified_evidence import (
    UnifiedEvidenceBundle,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
)


class RetrievalDomain(str, Enum):
    DOCUMENT = "document"
    ROBOT = "robot"


class RetrievalRankingError(ValueError):
    """Raised when ranked candidates violate the Day 12 protocol."""


class RetrievalBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_top_k: int = Field(ge=1)
    document_quota: int = Field(ge=0)
    robot_quota: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self):
        if (
            self.document_quota
            + self.robot_quota
            != self.total_top_k
        ):
            raise ValueError(
                "document_quota + robot_quota "
                "must equal total_top_k"
            )
        if self.document_quota < 1:
            raise ValueError(
                "Day 12 cross-domain budget requires "
                "document_quota >= 1"
            )
        if self.robot_quota < 1:
            raise ValueError(
                "Day 12 cross-domain budget requires "
                "robot_quota >= 1"
            )
        return self


DAY12_BASELINE_BUDGET = RetrievalBudget(
    total_top_k=5,
    document_quota=3,
    robot_quota=2,
)


class RankedEvidenceCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    domain: RetrievalDomain
    retriever_name: str = Field(min_length=1)
    rank: int = Field(ge=1)
    raw_score: float | None = None
    item: UnifiedEvidenceItem

    @model_validator(mode="after")
    def validate_domain_kind(self):
        if (
            self.raw_score is not None
            and not math.isfinite(self.raw_score)
        ):
            raise ValueError(
                "raw_score must be finite when present"
            )

        if (
            self.domain == RetrievalDomain.DOCUMENT
            and self.item.kind
            != UnifiedEvidenceKind.DOCUMENT_PAGE
        ):
            raise ValueError(
                "document candidate requires document_page item"
            )

        if (
            self.domain == RetrievalDomain.ROBOT
            and self.item.kind
            != UnifiedEvidenceKind.ROBOT_SAMPLE
        ):
            raise ValueError(
                "robot candidate requires robot_sample item"
            )

        return self


class RetrievalComposition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    bundle: UnifiedEvidenceBundle
    selected_candidates: list[RankedEvidenceCandidate]
    budget: RetrievalBudget


def validate_ranked_candidates(
    candidates: list[RankedEvidenceCandidate],
    *,
    domain: RetrievalDomain,
) -> list[RankedEvidenceCandidate]:
    if not candidates:
        raise RetrievalRankingError(
            f"{domain.value} candidate list must not be empty"
        )

    if any(
        candidate.domain != domain
        for candidate in candidates
    ):
        raise RetrievalRankingError(
            "candidate domain does not match list domain"
        )

    retriever_names = {
        candidate.retriever_name
        for candidate in candidates
    }
    if len(retriever_names) != 1:
        raise RetrievalRankingError(
            "one ranked list must come from one retriever"
        )

    expected_ranks = list(
        range(1, len(candidates) + 1)
    )
    actual_ranks = [
        candidate.rank
        for candidate in candidates
    ]
    if actual_ranks != expected_ranks:
        raise RetrievalRankingError(
            "candidate ranks must be contiguous "
            "and ordered from 1"
        )

    evidence_ids = [
        candidate.item.evidence_id
        for candidate in candidates
    ]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise RetrievalRankingError(
            "candidate evidence_id values must be unique"
        )

    return candidates


def compose_fixed_quota(
    *,
    query: str,
    document_candidates: list[RankedEvidenceCandidate],
    robot_candidates: list[RankedEvidenceCandidate],
    budget: RetrievalBudget = DAY12_BASELINE_BUDGET,
    bundle_id: str = "day12_retrieval_bundle",
) -> RetrievalComposition:
    """Compose a deterministic cross-domain bundle without score fusion.

    Cross-domain raw scores are intentionally never compared. Each source
    family keeps its own ranking semantics and contributes evidence under a
    fixed quota. Final bundle order is document rank order followed by robot
    rank order.
    """

    normalized_query = normalize_query(query)

    validate_ranked_candidates(
        document_candidates,
        domain=RetrievalDomain.DOCUMENT,
    )
    validate_ranked_candidates(
        robot_candidates,
        domain=RetrievalDomain.ROBOT,
    )

    selected_documents = document_candidates[
        : budget.document_quota
    ]
    selected_robot = robot_candidates[
        : budget.robot_quota
    ]

    selected = [
        *selected_documents,
        *selected_robot,
    ]

    if len(selected) > budget.total_top_k:
        raise RetrievalRankingError(
            "selected evidence exceeds total_top_k"
        )

    bundle = UnifiedEvidenceBundle(
        bundle_id=bundle_id,
        question=normalized_query,
        items=[
            candidate.item
            for candidate in selected
        ],
    )

    return RetrievalComposition(
        bundle=bundle,
        selected_candidates=selected,
        budget=budget,
    )
