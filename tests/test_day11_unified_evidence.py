from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    validate_cross_domain_bundle,
    validate_unified_citation_policy,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def vector(value: float) -> JointVector:
    return JointVector(
        shoulder_pan=value,
        shoulder_lift=value,
        elbow_flex=value,
        wrist_flex=value,
        wrist_roll=value,
        gripper=value,
    )


def document_item() -> UnifiedEvidenceItem:
    return UnifiedEvidenceItem(
        evidence_id="doc:manual:p1",
        kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
        refs=[
            EvidenceRef(
                source_id="manual",
                source_type=SourceType.PDF,
                page_number=1,
            )
        ],
        provenance=EvidenceProvenance(
            source_id="manual",
            source_type=SourceType.PDF,
            manifest_path="data/manifests/sources/manual.json",
            canonical_sha256=SHA_A,
        ),
        payload=DocumentPagePayload(
            page_number=1,
            text_sha256=SHA_B,
            char_count=10,
            text_excerpt="manual",
        ),
    )


def robot_item() -> UnifiedEvidenceItem:
    timestamp = 1.25
    refs = [
        EvidenceRef(
            source_id="ep0",
            source_type=SourceType.ROBOT_SEQUENCE,
            time_start_sec=timestamp,
            time_end_sec=timestamp,
            frame_index=5,
            camera=camera,
        )
        for camera in ("front", "wrist")
    ]
    cameras = [
        RobotCameraAsset(
            camera=camera,
            frame_index=5,
            timestamp_sec=timestamp,
            image_relpath=f"{camera}/000005.jpg",
            image_sha256=SHA_B if camera == "front" else SHA_C,
            source_timestamp_ns=1000,
            source_age_ms=1.0,
            width_px=640,
            height_px=480,
        )
        for camera in ("front", "wrist")
    ]
    snapshot = RobotStateActionSnapshot(
        frame_index=5,
        timestamp_sec=timestamp,
        observation=vector(1.0),
        action=vector(2.0),
        tracking_error=vector(1.0),
    )
    return UnifiedEvidenceItem(
        evidence_id="robot:ep0:f5",
        kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
        refs=refs,
        provenance=EvidenceProvenance(
            source_id="ep0",
            source_type=SourceType.ROBOT_SEQUENCE,
            manifest_path="data/manifests/robot_episodes/ep0.json",
            canonical_sha256=SHA_A,
            supporting_sha256={
                "metadata.json": SHA_B,
                "samples.csv": SHA_C,
            },
        ),
        payload=RobotSamplePayload(
            episode_id="ep0",
            frame_index=5,
            timestamp_sec=timestamp,
            cameras=cameras,
            state_action=snapshot,
        ),
    )


def bundle() -> UnifiedEvidenceBundle:
    return UnifiedEvidenceBundle(
        bundle_id="bundle0",
        question="test cross-domain evidence contract",
        items=[
            document_item(),
            robot_item(),
        ],
    )


def test_document_page_contract_accepts_pdf_ref():
    item = document_item()
    assert item.payload.page_number == 1
    assert item.refs[0].source_type == SourceType.PDF


def test_robot_sample_contract_accepts_paired_camera_refs():
    item = robot_item()
    assert [ref.camera for ref in item.refs] == [
        "front",
        "wrist",
    ]
    assert item.payload.state_action.frame_index == 5


def test_robot_sample_rejects_missing_camera_pair():
    value = robot_item().model_dump()
    value["refs"] = value["refs"][:1]
    with pytest.raises(ValidationError):
        UnifiedEvidenceItem.model_validate(value)


def test_robot_sample_rejects_ref_payload_frame_mismatch():
    value = robot_item().model_dump()
    value["refs"][0]["frame_index"] = 4
    with pytest.raises(ValidationError):
        UnifiedEvidenceItem.model_validate(value)


def test_bundle_rejects_duplicate_evidence_ids():
    first = document_item()
    second = robot_item().model_copy(
        update={"evidence_id": first.evidence_id}
    )
    with pytest.raises(ValidationError):
        UnifiedEvidenceBundle(
            bundle_id="duplicate",
            question="duplicate ids",
            items=[first, second],
        )


def test_cross_domain_validator_requires_both_kinds():
    valid, errors = validate_cross_domain_bundle(bundle())
    assert valid is True
    assert errors == []

    document_only = UnifiedEvidenceBundle(
        bundle_id="doc-only",
        question="doc only",
        items=[document_item()],
    )
    valid, errors = validate_cross_domain_bundle(document_only)
    assert valid is False
    assert errors == ["missing_robot_sample_evidence"]


def test_unified_citation_policy_accepts_supplied_refs():
    item_bundle = bundle()
    citations = [
        ref
        for item in item_bundle.items
        for ref in item.refs
    ]
    valid, errors = validate_unified_citation_policy(
        UnifiedGroundedAnswer(
            answer="supported",
            abstain=False,
            citations=citations,
        ),
        item_bundle,
    )
    assert valid is True
    assert errors == []


def test_unified_citation_policy_rejects_unsupported_ref():
    valid, errors = validate_unified_citation_policy(
        UnifiedGroundedAnswer(
            answer="unsupported",
            abstain=False,
            citations=[
                EvidenceRef(
                    source_id="other",
                    source_type=SourceType.PDF,
                    page_number=1,
                )
            ],
        ),
        bundle(),
    )
    assert valid is False
    assert any(
        error.startswith(
            "citation_outside_supplied_evidence="
        )
        for error in errors
    )
