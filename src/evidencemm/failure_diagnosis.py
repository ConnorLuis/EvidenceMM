from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evidencemm.grounding import required_fact_coverage
from evidencemm.schemas import EvidenceRef
from evidencemm.unified_evidence import (
    UnifiedEvidenceBundle,
    UnifiedGroundedAnswer,
    allowed_ref_keys,
    evidence_ref_key,
    validate_cross_domain_bundle,
    validate_unified_citation_policy,
)
from evidencemm.unified_grounding import (
    validate_required_citation_coverage,
)


class FailureStage(str, Enum):
    RETRIEVAL = "retrieval"
    EVIDENCE = "evidence"
    GENERATION = "generation"


class FailureCode(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"

    EVIDENCE_MISSING_DOCUMENT = "evidence_missing_document"
    EVIDENCE_MISSING_ROBOT = "evidence_missing_robot"
    EVIDENCE_MISSING_REQUIRED_REF = "evidence_missing_required_ref"

    GENERATION_HALLUCINATED_CITATION = (
        "generation_hallucinated_citation"
    )
    GENERATION_DUPLICATE_CITATION = (
        "generation_duplicate_citation"
    )
    GENERATION_CITATION_GAP = "generation_citation_gap"
    GENERATION_INCOMPLETE = "generation_incomplete"
    GENERATION_FALSE_ABSTENTION = (
        "generation_false_abstention"
    )
    GENERATION_OVERANSWER = "generation_overanswer"


class FailureSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class FailureFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: FailureStage
    code: FailureCode
    severity: FailureSeverity = FailureSeverity.ERROR
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class FailureDiagnosisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: list[FailureFinding] = Field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.findings

    @property
    def codes(self) -> list[str]:
        return [
            finding.code.value
            for finding in self.findings
        ]


def diagnose_retrieval_pages(
    *,
    ranked_pages: list[int],
    gold_pages: list[int],
    top_k: int,
) -> list[FailureFinding]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not gold_pages:
        raise ValueError("gold_pages must not be empty")
    if any(page < 1 for page in ranked_pages):
        raise ValueError("ranked pages must be >= 1")
    if len(set(ranked_pages)) != len(ranked_pages):
        raise ValueError("ranked pages must be unique")

    observed = ranked_pages[:top_k]
    gold = set(gold_pages)

    if any(page in gold for page in observed):
        return []

    return [
        FailureFinding(
            stage=FailureStage.RETRIEVAL,
            code=FailureCode.RETRIEVAL_MISS,
            message=(
                "No required document page is present "
                "within the evaluated retrieval cutoff."
            ),
            details={
                "top_k": top_k,
                "ranked_pages": observed,
                "gold_pages": gold_pages,
            },
        )
    ]


def diagnose_bundle(
    *,
    bundle: UnifiedEvidenceBundle,
    required_refs: list[EvidenceRef] | None = None,
    require_cross_domain: bool = True,
) -> list[FailureFinding]:
    findings: list[FailureFinding] = []

    if require_cross_domain:
        valid, errors = validate_cross_domain_bundle(bundle)
        if not valid:
            if "missing_document_page_evidence" in errors:
                findings.append(
                    FailureFinding(
                        stage=FailureStage.EVIDENCE,
                        code=(
                            FailureCode
                            .EVIDENCE_MISSING_DOCUMENT
                        ),
                        message=(
                            "The evidence bundle is missing "
                            "document-page evidence."
                        ),
                    )
                )
            if "missing_robot_sample_evidence" in errors:
                findings.append(
                    FailureFinding(
                        stage=FailureStage.EVIDENCE,
                        code=(
                            FailureCode
                            .EVIDENCE_MISSING_ROBOT
                        ),
                        message=(
                            "The evidence bundle is missing "
                            "robot-sample evidence."
                        ),
                    )
                )

    if required_refs:
        allowed = allowed_ref_keys(bundle)
        missing = [
            ref
            for ref in required_refs
            if evidence_ref_key(ref) not in allowed
        ]
        if missing:
            findings.append(
                FailureFinding(
                    stage=FailureStage.EVIDENCE,
                    code=(
                        FailureCode
                        .EVIDENCE_MISSING_REQUIRED_REF
                    ),
                    message=(
                        "One or more task-required evidence refs "
                        "are absent from the supplied bundle."
                    ),
                    details={
                        "missing_refs": [
                            ref.model_dump(
                                mode="json",
                                exclude_none=True,
                            )
                            for ref in missing
                        ]
                    },
                )
            )

    return findings


def diagnose_generation(
    *,
    answer: UnifiedGroundedAnswer,
    bundle: UnifiedEvidenceBundle,
    required_refs: list[EvidenceRef] | None = None,
    required_fact_groups: list[list[str]] | None = None,
    expected_answerable: bool | None = None,
) -> list[FailureFinding]:
    findings: list[FailureFinding] = []

    citation_valid, citation_errors = (
        validate_unified_citation_policy(
            answer,
            bundle,
        )
    )
    if not citation_valid:
        for error in citation_errors:
            if error.startswith(
                "citation_outside_supplied_evidence="
            ):
                findings.append(
                    FailureFinding(
                        stage=FailureStage.GENERATION,
                        code=(
                            FailureCode
                            .GENERATION_HALLUCINATED_CITATION
                        ),
                        message=(
                            "The generated answer cites evidence "
                            "that was not supplied to generation."
                        ),
                        details={"validator_error": error},
                    )
                )
            elif error == "duplicate_citations":
                findings.append(
                    FailureFinding(
                        stage=FailureStage.GENERATION,
                        code=(
                            FailureCode
                            .GENERATION_DUPLICATE_CITATION
                        ),
                        message=(
                            "The generated answer contains "
                            "duplicate citations."
                        ),
                    )
                )
            elif error == (
                "answerable_response_requires_citation"
            ):
                findings.append(
                    FailureFinding(
                        stage=FailureStage.GENERATION,
                        code=(
                            FailureCode
                            .GENERATION_CITATION_GAP
                        ),
                        message=(
                            "A non-abstaining generated answer "
                            "has no supporting citation."
                        ),
                    )
                )

    if required_refs:
        covered, errors = (
            validate_required_citation_coverage(
                answer,
                required_refs,
            )
        )
        if not covered:
            findings.append(
                FailureFinding(
                    stage=FailureStage.GENERATION,
                    code=(
                        FailureCode
                        .GENERATION_CITATION_GAP
                    ),
                    message=(
                        "The generated answer omits one or more "
                        "required supporting citations."
                    ),
                    details={
                        "validator_errors": errors
                    },
                )
            )

    if required_fact_groups:
        coverage = required_fact_coverage(
            answer.answer,
            required_fact_groups,
        )
        if coverage < 1.0:
            findings.append(
                FailureFinding(
                    stage=FailureStage.GENERATION,
                    code=(
                        FailureCode
                        .GENERATION_INCOMPLETE
                    ),
                    message=(
                        "The generated answer does not cover all "
                        "required fact groups."
                    ),
                    details={
                        "required_fact_coverage": coverage
                    },
                )
            )

    if expected_answerable is True and answer.abstain:
        findings.append(
            FailureFinding(
                stage=FailureStage.GENERATION,
                code=(
                    FailureCode
                    .GENERATION_FALSE_ABSTENTION
                ),
                message=(
                    "Generation abstained although the "
                    "evaluation fixture is answerable."
                ),
            )
        )

    if expected_answerable is False and not answer.abstain:
        findings.append(
            FailureFinding(
                stage=FailureStage.GENERATION,
                code=FailureCode.GENERATION_OVERANSWER,
                message=(
                    "Generation answered although the "
                    "evaluation fixture requires abstention."
                ),
            )
        )

    # Multiple validator paths can identify the same citation-gap failure.
    deduplicated: list[FailureFinding] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        key = (
            finding.stage.value,
            finding.code.value,
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(finding)

    return deduplicated


def diagnose_pipeline(
    *,
    retrieval_findings: list[FailureFinding] | None = None,
    evidence_findings: list[FailureFinding] | None = None,
    generation_findings: list[FailureFinding] | None = None,
) -> FailureDiagnosisReport:
    findings = [
        *(retrieval_findings or []),
        *(evidence_findings or []),
        *(generation_findings or []),
    ]

    stage_order = {
        FailureStage.RETRIEVAL: 0,
        FailureStage.EVIDENCE: 1,
        FailureStage.GENERATION: 2,
    }
    findings.sort(
        key=lambda item: (
            stage_order[item.stage],
            item.code.value,
        )
    )
    return FailureDiagnosisReport(
        findings=findings
    )
