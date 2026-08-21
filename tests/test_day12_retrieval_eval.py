from __future__ import annotations

from evidencemm.retrieval_eval import (
    evaluate_document_candidates,
    robot_profile_matches,
    validate_robot_candidate_evidence,
)
from evidencemm.retrieval_ranking import (
    RankedEvidenceCandidate,
    RetrievalDomain,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalQueryProfile,
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


def document_candidate(
    rank: int,
    page_number: int,
) -> RankedEvidenceCandidate:
    item = UnifiedEvidenceItem(
        evidence_id=f"doc:p{page_number}",
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
            char_count=1,
            text_excerpt="x",
        ),
    )
    return RankedEvidenceCandidate(
        domain=RetrievalDomain.DOCUMENT,
        retriever_name="bm25",
        rank=rank,
        raw_score=float(10 - rank),
        item=item,
    )


def robot_candidate() -> RankedEvidenceCandidate:
    timestamp = 1.0
    frame_index = 15
    cameras = [
        RobotCameraAsset(
            camera=camera,
            frame_index=frame_index,
            timestamp_sec=timestamp,
            image_relpath=f"{camera}/000015.jpg",
            image_sha256=SHA,
            source_timestamp_ns=1000,
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
    item = UnifiedEvidenceItem(
        evidence_id="robot:f15",
        kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
        refs=[
            EvidenceRef(
                source_id="episode",
                source_type=SourceType.ROBOT_SEQUENCE,
                time_start_sec=timestamp,
                time_end_sec=timestamp,
                frame_index=frame_index,
                camera=camera,
            )
            for camera in ("front", "wrist")
        ],
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
    return RankedEvidenceCandidate(
        domain=RetrievalDomain.ROBOT,
        retriever_name="robot",
        rank=1,
        raw_score=1.0,
        item=item,
    )


def test_document_metrics_rank_one():
    metrics = evaluate_document_candidates(
        candidates=[
            document_candidate(1, 3),
            document_candidate(2, 5),
        ],
        gold_pages=[3],
        candidate_pool_k=5,
    )
    assert metrics.gold_rank == 1
    assert metrics.hit_at_1 is True
    assert metrics.hit_at_3 is True
    assert metrics.reciprocal_rank == 1.0


def test_document_metrics_rank_four():
    metrics = evaluate_document_candidates(
        candidates=[
            document_candidate(1, 1),
            document_candidate(2, 2),
            document_candidate(3, 3),
            document_candidate(4, 4),
        ],
        gold_pages=[4],
        candidate_pool_k=5,
    )
    assert metrics.gold_rank == 4
    assert metrics.hit_at_1 is False
    assert metrics.hit_at_3 is False
    assert metrics.hit_at_5 is True
    assert metrics.reciprocal_rank == 0.25


def test_document_metrics_missing_gold():
    metrics = evaluate_document_candidates(
        candidates=[
            document_candidate(1, 1),
            document_candidate(2, 2),
        ],
        gold_pages=[8],
        candidate_pool_k=5,
    )
    assert metrics.gold_rank is None
    assert metrics.hit_at_5 is False
    assert metrics.reciprocal_rank == 0.0


def test_robot_profile_exact_match():
    profile = RobotSignalQueryProfile(
        joints=("gripper",),
        signals=("action",),
        explicit_joint_terms=True,
        explicit_signal_terms=True,
    )
    assert robot_profile_matches(
        profile=profile,
        expected_joints=["gripper"],
        expected_signals=["action"],
    )


def test_robot_profile_rejects_fallback_profile():
    profile = RobotSignalQueryProfile(
        joints=("gripper",),
        signals=("action",),
        explicit_joint_terms=False,
        explicit_signal_terms=True,
    )
    assert not robot_profile_matches(
        profile=profile,
        expected_joints=["gripper"],
        expected_signals=["action"],
    )


def test_robot_candidate_evidence_valid():
    valid, errors = validate_robot_candidate_evidence(
        [robot_candidate()]
    )
    assert valid is True
    assert errors == []
