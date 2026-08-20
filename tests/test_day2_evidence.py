from __future__ import annotations

from datetime import datetime, timezone

import pytest
from PIL import Image
from pydantic import ValidationError

from evidencemm.data_binding import bind_source
from evidencemm.schemas import (
    EvalCase,
    EvidenceRef,
    NormalizedBBox,
    SourceManifest,
    SourceType,
)


def test_pdf_evidence_requires_page_number():
    with pytest.raises(ValidationError):
        EvidenceRef(
            source_id="manual",
            source_type=SourceType.PDF,
        )


def test_source_manifest_requires_pdf_page_count():
    with pytest.raises(ValidationError):
        SourceManifest(
            source_id="manual",
            source_type=SourceType.PDF,
            local_path="data/raw/manual.pdf",
            sha256="0" * 64,
            size_bytes=100,
            added_at=datetime.now(timezone.utc),
        )


def test_verified_case_requires_traceable_evidence_and_verifier():
    evidence = EvidenceRef(
        source_id="img",
        source_type=SourceType.IMAGE,
        bbox=NormalizedBBox(
            x1=0.0,
            y1=0.0,
            x2=1.0,
            y2=1.0,
        ),
    )

    with pytest.raises(ValidationError):
        EvalCase(
            case_id="v1",
            question="q",
            input_ids=["img"],
            answerable=True,
            expected_answer="a",
            expected_evidence=[evidence],
            annotation_status="verified",
        )


def test_bind_image_uses_repo_relative_path(tmp_path):
    repo = tmp_path / "repo"
    image_path = repo / "data" / "raw" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image_path)

    manifest = bind_source(
        source_id="sample_image",
        source_type=SourceType.IMAGE,
        path=image_path,
        project_root=repo,
    )

    assert manifest.local_path == "data/raw/sample.jpg"


def test_verified_abstention_case_is_valid():
    evidence = EvidenceRef(
        source_id="img",
        source_type=SourceType.IMAGE,
        bbox=NormalizedBBox(
            x1=0.0,
            y1=0.0,
            x2=1.0,
            y2=1.0,
        ),
    )

    case = EvalCase(
        case_id="abstain-1",
        question="Can this be determined?",
        input_ids=["img"],
        answerable=False,
        expected_answer="Insufficient evidence.",
        expected_evidence=[evidence],
        annotation_status="verified",
        verified_by="human",
        verified_at=datetime.now(timezone.utc),
    )

    assert case.answerable is False
