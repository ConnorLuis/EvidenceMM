from __future__ import annotations

import json

import pytest

from evidencemm.retrieval_grounded_generation import (
    CompactGroundedAnswer,
    build_citation_alias_map,
    count_visual_inputs,
    dynamic_robot_fact_groups,
    find_document_page_item,
    required_citation_aliases,
    required_generation_refs,
    resolve_compact_grounded_answer,
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
            page_image_path=(
                f"data/processed/vision/p{page_number}.png"
            ),
        ),
    )


def robot_item(
    frame_index: int,
    timestamp: float,
) -> UnifiedEvidenceItem:
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
    return UnifiedEvidenceItem(
        evidence_id=f"robot:f{frame_index}",
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
            state_action=RobotStateActionSnapshot(
                frame_index=frame_index,
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
            document_item(3),
            document_item(5),
            document_item(8),
            robot_item(155, 10.3383813),
            robot_item(156, 10.4053413),
        ],
    )


def test_find_required_document_page():
    item = find_document_page_item(
        bundle(),
        page_number=3,
    )
    assert item is not None
    assert item.payload.page_number == 3


def test_missing_required_document_page_returns_none():
    assert (
        find_document_page_item(
            bundle(),
            page_number=4,
        )
        is None
    )


def test_required_generation_refs_are_one_doc_plus_four_robot():
    refs = required_generation_refs(
        bundle=bundle(),
        document_page=3,
    )
    assert len(refs) == 5
    assert refs[0].page_number == 3
    assert [
        ref.camera
        for ref in refs[1:]
    ] == [
        "front",
        "wrist",
        "front",
        "wrist",
    ]


def test_dynamic_robot_fact_groups_accept_chinese_copula_form():
    groups = dynamic_robot_fact_groups(
        bundle()
    )
    assert len(groups) == 4
    assert "frame_index 是 155" in groups[0]
    assert "10.3383813" in groups[1]
    assert "frame_index 是 156" in groups[2]
    assert "10.4053413" in groups[3]


def test_visual_input_count_is_three_pages_plus_four_robot_images():
    assert count_visual_inputs(bundle()) == 7


def test_citation_alias_map_is_deterministic():
    aliases = build_citation_alias_map(
        bundle()
    )
    assert list(aliases) == [
        "DOC_P3",
        "DOC_P5",
        "DOC_P8",
        "ROBOT_F155_FRONT",
        "ROBOT_F155_WRIST",
        "ROBOT_F156_FRONT",
        "ROBOT_F156_WRIST",
    ]


def test_compact_citations_resolve_back_to_evidence_refs():
    aliases = build_citation_alias_map(
        bundle()
    )
    compact = CompactGroundedAnswer(
        answer="ok",
        abstain=False,
        citation_ids=[
            "DOC_P3",
            "ROBOT_F155_FRONT",
            "ROBOT_F155_WRIST",
            "ROBOT_F156_FRONT",
            "ROBOT_F156_WRIST",
        ],
    )
    answer = resolve_compact_grounded_answer(
        compact,
        aliases,
    )
    assert len(answer.citations) == 5
    assert answer.citations[0].page_number == 3
    assert answer.citations[1].frame_index == 155
    assert answer.citations[-1].camera == "wrist"


def test_compact_citation_duplicate_is_rejected():
    aliases = build_citation_alias_map(
        bundle()
    )
    compact = CompactGroundedAnswer(
        answer="bad",
        abstain=False,
        citation_ids=[
            "DOC_P3",
            "DOC_P3",
        ],
    )
    with pytest.raises(
        ValueError,
        match="duplicate compact citation ids",
    ):
        resolve_compact_grounded_answer(
            compact,
            aliases,
        )


def test_required_aliases_are_one_doc_plus_four_robot():
    b = bundle()
    aliases = build_citation_alias_map(b)
    refs = required_generation_refs(
        bundle=b,
        document_page=3,
    )
    required = required_citation_aliases(
        aliases=aliases,
        required_refs=refs,
    )
    assert required == [
        "DOC_P3",
        "ROBOT_F155_FRONT",
        "ROBOT_F155_WRIST",
        "ROBOT_F156_FRONT",
        "ROBOT_F156_WRIST",
    ]
