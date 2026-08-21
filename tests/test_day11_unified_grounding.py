from __future__ import annotations

import json

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
from evidencemm.unified_grounding import (
    allowed_citations_json,
    build_unified_messages,
    parse_unified_grounded_answer,
    validate_required_citation_coverage,
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


def fixture_bundle() -> UnifiedEvidenceBundle:
    document_ref = EvidenceRef(
        source_id="manual",
        source_type=SourceType.PDF,
        page_number=1,
    )
    robot_refs = [
        EvidenceRef(
            source_id="ep0",
            source_type=SourceType.ROBOT_SEQUENCE,
            time_start_sec=1.0,
            time_end_sec=1.0,
            frame_index=15,
            camera=camera,
        )
        for camera in ("front", "wrist")
    ]

    return UnifiedEvidenceBundle(
        bundle_id="b0",
        question=(
            "报告页码、frame、时间、相机以及是否包含 "
            "observation/action。"
        ),
        items=[
            UnifiedEvidenceItem(
                evidence_id="doc",
                kind=UnifiedEvidenceKind.DOCUMENT_PAGE,
                refs=[document_ref],
                provenance=EvidenceProvenance(
                    source_id="manual",
                    source_type=SourceType.PDF,
                    manifest_path="manual.json",
                    canonical_sha256=SHA,
                ),
                payload=DocumentPagePayload(
                    page_number=1,
                    text_sha256=SHA,
                    char_count=10,
                    text_excerpt="STS3215",
                    page_image_path="page.png",
                ),
            ),
            UnifiedEvidenceItem(
                evidence_id="robot",
                kind=UnifiedEvidenceKind.ROBOT_SAMPLE,
                refs=robot_refs,
                provenance=EvidenceProvenance(
                    source_id="ep0",
                    source_type=SourceType.ROBOT_SEQUENCE,
                    manifest_path="episode.json",
                    canonical_sha256=SHA,
                ),
                payload=RobotSamplePayload(
                    episode_id="ep0",
                    frame_index=15,
                    timestamp_sec=1.0,
                    cameras=[
                        RobotCameraAsset(
                            camera=camera,
                            frame_index=15,
                            timestamp_sec=1.0,
                            image_relpath=(
                                f"{camera}/000015.jpg"
                            ),
                            image_sha256=SHA,
                            source_timestamp_ns=1000,
                            source_age_ms=1.0,
                            width_px=640,
                            height_px=480,
                        )
                        for camera in ("front", "wrist")
                    ],
                    state_action=RobotStateActionSnapshot(
                        frame_index=15,
                        timestamp_sec=1.0,
                        observation=vector(1.0),
                        action=vector(2.0),
                        tracking_error=vector(1.0),
                    ),
                ),
            ),
        ],
    )


def test_parse_unified_grounded_answer_from_json():
    text = json.dumps(
        {
            "answer": "supported",
            "abstain": False,
            "citations": [
                {
                    "source_id": "manual",
                    "source_type": "pdf",
                    "page_number": 1,
                }
            ],
        }
    )
    answer = parse_unified_grounded_answer(text)
    assert answer.abstain is False
    assert answer.citations[0].page_number == 1


def test_parse_unified_grounded_answer_from_fenced_json():
    text = """```json
{"answer":"supported","abstain":false,"citations":[]}
```"""
    answer = parse_unified_grounded_answer(text)
    assert answer.answer == "supported"


def test_allowed_citations_preserve_all_cross_domain_refs():
    allowed = allowed_citations_json(
        fixture_bundle()
    )
    assert len(allowed) == 3
    assert allowed[0]["source_type"] == "pdf"
    assert [
        item.get("camera")
        for item in allowed[1:]
    ] == ["front", "wrist"]


def test_required_citation_coverage_requires_every_fixture_ref():
    bundle = fixture_bundle()
    all_refs = [
        ref
        for item in bundle.items
        for ref in item.refs
    ]
    answer = UnifiedGroundedAnswer(
        answer="supported",
        abstain=False,
        citations=all_refs[:-1],
    )
    valid, errors = (
        validate_required_citation_coverage(
            answer,
            all_refs,
        )
    )
    assert valid is False
    assert errors[0].startswith(
        "missing_required_citations="
    )


def test_build_messages_contains_three_visual_inputs():
    bundle = fixture_bundle()
    messages = build_unified_messages(
        bundle=bundle,
        project_root="/project",
        episode_dir="/episode",
    )
    user_content = messages[1]["content"]
    images = [
        item
        for item in user_content
        if item["type"] == "image"
    ]
    assert len(images) == 3
    assert images[0]["image"].endswith("page.png")
    assert images[1]["image"].endswith(
        "front/000015.jpg"
    )
    assert images[2]["image"].endswith(
        "wrist/000015.jpg"
    )
