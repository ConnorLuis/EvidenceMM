from __future__ import annotations

from evidencemm.failure_diagnosis import (
    FailureCode,
    diagnose_bundle,
    diagnose_generation,
    diagnose_pipeline,
    diagnose_retrieval_pages,
)
from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.state_action_selection import JointVector
from evidencemm.unified_evidence import (
    DocumentPagePayload,
    EvidenceProvenance,
    RobotCameraAsset,
    RobotSamplePayload,
    RobotStateActionSnapshot,
    UnifiedEvidenceBundle,
    UnifiedEvidenceItem,
    UnifiedEvidenceKind,
    UnifiedGroundedAnswer,
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


def doc_item(page: int) -> UnifiedEvidenceItem:
    return UnifiedEvidenceItem(
        evidence_id=f"doc:p{page}",
        kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
        refs=[
            EvidenceRef(
                source_id="manual",
                source_type=SourceType.PDF,
                page_number=page,
            )
        ],
        provenance=EvidenceProvenance(
            source_id="manual",
            source_type=SourceType.PDF,
            manifest_path="manual.json",
            canonical_sha256=SHA,
        ),
        payload=DocumentPagePayload(
            page_number=page,
            text_sha256=SHA,
            char_count=1,
            text_excerpt="x",
        ),
    )


def robot_item(frame: int) -> UnifiedEvidenceItem:
    timestamp = frame / 10.0
    cameras = [
        RobotCameraAsset(
            camera=camera,
            frame_index=frame,
            timestamp_sec=timestamp,
            image_relpath=f"{camera}/{frame}.jpg",
            image_sha256=SHA,
            source_timestamp_ns=1000 + frame,
            source_age_ms=1.0,
            width_px=640,
            height_px=480,
        )
        for camera in ("front", "wrist")
    ]
    return UnifiedEvidenceItem(
        evidence_id=f"robot:f{frame}",
        kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
        refs=[
            EvidenceRef(
                source_id="episode",
                source_type=SourceType.ROBOT_SEQUENCE,
                time_start_sec=timestamp,
                time_end_sec=timestamp,
                frame_index=frame,
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
            frame_index=frame,
            timestamp_sec=timestamp,
            cameras=cameras,
            state_action=RobotStateActionSnapshot(
                frame_index=frame,
                timestamp_sec=timestamp,
                observation=vector(1.0),
                action=vector(2.0),
                tracking_error=vector(1.0),
            ),
        ),
    )


def bundle() -> UnifiedEvidenceBundle:
    return UnifiedEvidenceBundle(
        bundle_id="b",
        question="q",
        items=[
            doc_item(3),
            robot_item(155),
        ],
    )


def required_refs():
    b = bundle()
    return [
        ref
        for item in b.items
        for ref in item.refs
    ]


def codes(findings):
    return {
        finding.code
        for finding in findings
    }


def test_retrieval_miss_detected():
    findings = diagnose_retrieval_pages(
        ranked_pages=[8, 5, 2],
        gold_pages=[3],
        top_k=3,
    )
    assert FailureCode.RETRIEVAL_MISS in codes(
        findings
    )


def test_retrieval_hit_is_healthy():
    assert not diagnose_retrieval_pages(
        ranked_pages=[8, 3, 5],
        gold_pages=[3],
        top_k=3,
    )


def test_missing_robot_evidence_detected():
    docs_only = UnifiedEvidenceBundle(
        bundle_id="docs",
        question="q",
        items=[doc_item(3)],
    )
    findings = diagnose_bundle(
        bundle=docs_only,
        require_cross_domain=True,
    )
    assert (
        FailureCode.EVIDENCE_MISSING_ROBOT
        in codes(findings)
    )


def test_missing_required_ref_detected():
    b = bundle()
    missing = EvidenceRef(
        source_id="manual",
        source_type=SourceType.PDF,
        page_number=4,
    )
    findings = diagnose_bundle(
        bundle=b,
        required_refs=[missing],
    )
    assert (
        FailureCode.EVIDENCE_MISSING_REQUIRED_REF
        in codes(findings)
    )


def test_hallucinated_citation_detected():
    b = bundle()
    outside = EvidenceRef(
        source_id="manual",
        source_type=SourceType.PDF,
        page_number=999,
    )
    answer = UnifiedGroundedAnswer(
        answer="6V",
        abstain=False,
        citations=[
            b.items[0].refs[0],
            outside,
        ],
    )
    findings = diagnose_generation(
        answer=answer,
        bundle=b,
    )
    assert (
        FailureCode.GENERATION_HALLUCINATED_CITATION
        in codes(findings)
    )


def test_duplicate_citation_detected():
    b = bundle()
    ref = b.items[0].refs[0]
    answer = UnifiedGroundedAnswer(
        answer="6V",
        abstain=False,
        citations=[ref, ref],
    )
    findings = diagnose_generation(
        answer=answer,
        bundle=b,
    )
    assert (
        FailureCode.GENERATION_DUPLICATE_CITATION
        in codes(findings)
    )


def test_required_citation_gap_detected():
    b = bundle()
    answer = UnifiedGroundedAnswer(
        answer="6V front wrist",
        abstain=False,
        citations=b.items[0].refs,
    )
    findings = diagnose_generation(
        answer=answer,
        bundle=b,
        required_refs=required_refs(),
    )
    assert (
        FailureCode.GENERATION_CITATION_GAP
        in codes(findings)
    )


def test_incomplete_generation_detected():
    b = bundle()
    answer = UnifiedGroundedAnswer(
        answer="6V observation",
        abstain=False,
        citations=required_refs(),
    )
    findings = diagnose_generation(
        answer=answer,
        bundle=b,
        required_fact_groups=[
            ["6V"],
            ["action"],
        ],
    )
    assert (
        FailureCode.GENERATION_INCOMPLETE
        in codes(findings)
    )


def test_false_abstention_detected():
    b = bundle()
    answer = UnifiedGroundedAnswer(
        answer="insufficient",
        abstain=True,
        citations=[],
    )
    findings = diagnose_generation(
        answer=answer,
        bundle=b,
        expected_answerable=True,
    )
    assert (
        FailureCode.GENERATION_FALSE_ABSTENTION
        in codes(findings)
    )


def test_overanswer_detected():
    b = bundle()
    answer = UnifiedGroundedAnswer(
        answer="answer",
        abstain=False,
        citations=b.items[0].refs,
    )
    findings = diagnose_generation(
        answer=answer,
        bundle=b,
        expected_answerable=False,
    )
    assert (
        FailureCode.GENERATION_OVERANSWER
        in codes(findings)
    )


def test_healthy_pipeline_has_no_findings():
    b = bundle()
    refs = required_refs()
    answer = UnifiedGroundedAnswer(
        answer=(
            "6V front wrist observation action"
        ),
        abstain=False,
        citations=refs,
    )
    report = diagnose_pipeline(
        retrieval_findings=(
            diagnose_retrieval_pages(
                ranked_pages=[3, 4, 5],
                gold_pages=[3],
                top_k=3,
            )
        ),
        evidence_findings=diagnose_bundle(
            bundle=b,
            required_refs=refs,
        ),
        generation_findings=diagnose_generation(
            answer=answer,
            bundle=b,
            required_refs=refs,
            required_fact_groups=[
                ["6V"],
                ["front"],
                ["wrist"],
                ["observation"],
                ["action"],
            ],
            expected_answerable=True,
        ),
    )
    assert report.healthy is True
    assert report.codes == []
