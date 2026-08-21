from __future__ import annotations

import pytest
from pydantic import ValidationError

from evidencemm.retrieval_ranking import (
    DAY12_BASELINE_BUDGET,
    RankedEvidenceCandidate,
    RetrievalBudget,
    RetrievalDomain,
    RetrievalRankingError,
    compose_fixed_quota,
    validate_ranked_candidates,
)
from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.state_action_selection import JointVector
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    EvidenceProvenance,
    RobotCameraAsset,
    RobotSamplePayload,
    RobotStateActionSnapshot,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
)


SHA = "a" * 64


def vector(value: float) -> JointVector:
    return JointVector(
        shoulder_pan=value,
        shoulder_lift=value,
        elbow_flex=value,
        wrist_flex=value,
        wrist_roll=value,
        gripper=value,
    )


def document_item(
    page_number: int,
) -> UnifiedEvidenceItem:
    return UnifiedEvidenceItem(
        evidence_id=f"doc:manual:p{page_number}",
        kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
        refs=[
            EvidenceRef(
                source_id="manual",
                source_type=SourceType.PDF,
                page_number=page_number,
            )
        ],
        provenance=EvidenceProvenance(
            source_id="manual",
            source_type=SourceType.PDF,
            manifest_path="manual.json",
            canonical_sha256=SHA,
        ),
        payload=DocumentPagePayload(
            page_number=page_number,
            text_sha256=SHA,
            char_count=10,
            text_excerpt=f"page {page_number}",
        ),
    )


def robot_item(
    frame_index: int,
) -> UnifiedEvidenceItem:
    timestamp = frame_index / 10.0
    refs = [
        EvidenceRef(
            source_id="episode",
            source_type=SourceType.ROBOT_SEQUENCE,
            time_start_sec=timestamp,
            time_end_sec=timestamp,
            frame_index=frame_index,
            camera=camera,
        )
        for camera in ("front", "wrist")
    ]
    cameras = [
        RobotCameraAsset(
            camera=camera,
            frame_index=frame_index,
            timestamp_sec=timestamp,
            image_relpath=(
                f"{camera}/{frame_index:06d}.jpg"
            ),
            image_sha256=SHA,
            source_timestamp_ns=1000 + frame_index,
            source_age_ms=1.0,
            width_px=640,
            height_px=480,
        )
        for camera in ("front", "wrist")
    ]
    snapshot = RobotStateActionSnapshot(
        frame_index=frame_index,
        timestamp_sec=timestamp,
        observation=vector(1.0),
        action=vector(2.0),
        tracking_error=vector(1.0),
    )
    return UnifiedEvidenceItem(
        evidence_id=f"robot:episode:f{frame_index}",
        kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
        refs=refs,
        provenance=EvidenceProvenance(
            source_id="episode",
            source_type=SourceType.ROBOT_SEQUENCE,
            manifest_path="episode.json",
            canonical_sha256=SHA,
        ),
        payload=RobotSamplePayload(
            episode_id="episode",
            frame_index=frame_index,
            timestamp_sec=timestamp,
            cameras=cameras,
            state_action=snapshot,
        ),
    )


def document_candidates(
    count: int,
) -> list[RankedEvidenceCandidate]:
    return [
        RankedEvidenceCandidate(
            domain=RetrievalDomain.DOCUMENT,
            retriever_name="bm25",
            rank=rank,
            raw_score=100.0 - rank,
            item=document_item(rank),
        )
        for rank in range(1, count + 1)
    ]


def robot_candidates(
    count: int,
) -> list[RankedEvidenceCandidate]:
    return [
        RankedEvidenceCandidate(
            domain=RetrievalDomain.ROBOT,
            retriever_name="robot_baseline",
            rank=rank,
            raw_score=0.01 * rank,
            item=robot_item(rank),
        )
        for rank in range(1, count + 1)
    ]


def test_day12_budget_is_frozen_to_three_plus_two():
    assert DAY12_BASELINE_BUDGET.total_top_k == 5
    assert DAY12_BASELINE_BUDGET.document_quota == 3
    assert DAY12_BASELINE_BUDGET.robot_quota == 2


def test_budget_requires_exact_cross_domain_total():
    with pytest.raises(ValidationError):
        RetrievalBudget(
            total_top_k=5,
            document_quota=3,
            robot_quota=1,
        )


def test_candidate_domain_must_match_item_kind():
    with pytest.raises(ValidationError):
        RankedEvidenceCandidate(
            domain=RetrievalDomain.ROBOT,
            retriever_name="bad",
            rank=1,
            raw_score=1.0,
            item=document_item(1),
        )


def test_ranked_list_requires_contiguous_ordered_ranks():
    candidates = document_candidates(2)
    candidates[1] = candidates[1].model_copy(
        update={"rank": 3}
    )

    with pytest.raises(
        RetrievalRankingError,
        match="contiguous",
    ):
        validate_ranked_candidates(
            candidates,
            domain=RetrievalDomain.DOCUMENT,
        )


def test_ranked_list_requires_one_retriever():
    candidates = document_candidates(2)
    candidates[1] = candidates[1].model_copy(
        update={"retriever_name": "other"}
    )

    with pytest.raises(
        RetrievalRankingError,
        match="one retriever",
    ):
        validate_ranked_candidates(
            candidates,
            domain=RetrievalDomain.DOCUMENT,
        )


def test_composer_selects_three_documents_and_two_robot_items():
    result = compose_fixed_quota(
        query="  cross-domain query  ",
        document_candidates=document_candidates(5),
        robot_candidates=robot_candidates(4),
    )

    assert result.bundle.question == (
        "cross-domain query"
    )
    assert len(result.bundle.items) == 5
    assert [
        item.kind
        for item in result.bundle.items
    ] == [
        UnifiedEvidenceKind.DOCUMENT_PAGE,
        UnifiedEvidenceKind.DOCUMENT_PAGE,
        UnifiedEvidenceKind.DOCUMENT_PAGE,
        UnifiedEvidenceKind.ROBOT_SAMPLE,
        UnifiedEvidenceKind.ROBOT_SAMPLE,
    ]


def test_cross_domain_raw_scores_are_not_compared():
    documents = document_candidates(3)
    robots = robot_candidates(2)

    documents[0] = documents[0].model_copy(
        update={"raw_score": -1_000_000.0}
    )
    robots[1] = robots[1].model_copy(
        update={"raw_score": 1_000_000.0}
    )

    result = compose_fixed_quota(
        query="query",
        document_candidates=documents,
        robot_candidates=robots,
    )

    assert [
        candidate.rank
        for candidate in result.selected_candidates
    ] == [1, 2, 3, 1, 2]
    assert [
        candidate.domain
        for candidate in result.selected_candidates
    ] == [
        RetrievalDomain.DOCUMENT,
        RetrievalDomain.DOCUMENT,
        RetrievalDomain.DOCUMENT,
        RetrievalDomain.ROBOT,
        RetrievalDomain.ROBOT,
    ]


def test_composer_allows_short_source_lists_without_filler():
    result = compose_fixed_quota(
        query="query",
        document_candidates=document_candidates(2),
        robot_candidates=robot_candidates(1),
    )

    assert len(result.bundle.items) == 3
    assert len(result.selected_candidates) == 3


def test_empty_source_family_is_rejected_for_cross_domain_baseline():
    with pytest.raises(
        RetrievalRankingError,
        match="robot candidate list",
    ):
        compose_fixed_quota(
            query="query",
            document_candidates=document_candidates(3),
            robot_candidates=[],
        )
