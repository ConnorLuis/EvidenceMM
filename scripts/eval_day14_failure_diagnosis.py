from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.document_candidate_retrieval import (
    DocumentBM25CandidateRetriever,
)
from evidencemm.failure_diagnosis import (
    diagnose_bundle,
    diagnose_generation,
    diagnose_pipeline,
    diagnose_retrieval_pages,
)
from evidencemm.retrieval_ranking import (
    DAY12_BASELINE_BUDGET,
    compose_fixed_quota,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalCandidateRetriever,
)
from evidencemm.schemas import EvidenceRef, SourceType
from evidencemm.unified_evidence import (
    UnifiedEvidenceBundle,
    UnifiedEvidenceKind,
    UnifiedGroundedAnswer,
)


ROOT = Path(__file__).resolve().parents[1]


def load_expected(
    path: Path,
) -> dict[str, list[str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return {
        row["scenario_id"]: row["expected_codes"]
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-dir",
        required=True,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/day14_failure_diagnosis.yaml"
        ).read_text(encoding="utf-8")
    )
    retrieval_config = yaml.safe_load(
        (
            ROOT
            / config["retrieval_config"]
        ).read_text(encoding="utf-8")
    )
    robot_config = yaml.safe_load(
        (
            ROOT
            / config["robot_config"]
        ).read_text(encoding="utf-8")
    )

    candidate_pool_k = int(
        config["candidate_pool_k"]
    )
    if candidate_pool_k != 5:
        raise ValueError(
            "Day 14 candidate_pool_k must remain 5"
        )

    query = str(
        retrieval_config["query"]
    ).strip()
    gold_page = int(
        config["document_gold_page"]
    )

    document_retriever = (
        DocumentBM25CandidateRetriever(
            project_root=ROOT,
            source_manifest_path=(
                retrieval_config[
                    "document_manifest"
                ]
            ),
            visual_manifest_path=(
                retrieval_config[
                    "document_visual_manifest"
                ]
            ),
        )
    )

    episode_id = str(
        retrieval_config["episode_id"]
    )
    episode_manifest_path = (
        ROOT
        / robot_config["manifest_root"]
        / f"{episode_id}.json"
    )
    frame_records_path = (
        ROOT
        / robot_config["processed_root"]
        / episode_id
        / "frames.jsonl"
    )

    robot_retriever = (
        RobotSignalCandidateRetriever(
            project_root=ROOT,
            episode_manifest_path=(
                episode_manifest_path
            ),
            episode_dir=args.episode_dir,
            frame_records_path=(
                frame_records_path
            ),
        )
    )

    document_candidates = (
        document_retriever.search(
            query,
            top_k=candidate_pool_k,
        )
    )
    robot_candidates = (
        robot_retriever.search(
            query,
            top_k=candidate_pool_k,
        )
    )

    composition = compose_fixed_quota(
        query=query,
        document_candidates=document_candidates,
        robot_candidates=robot_candidates,
        budget=DAY12_BASELINE_BUDGET,
        bundle_id="day14_failure_reference",
    )
    bundle = composition.bundle

    document_item = next(
        item
        for item in bundle.items
        if (
            item.kind
            == UnifiedEvidenceKind.DOCUMENT_PAGE
            and item.payload.page_number == gold_page
        )
    )
    robot_items = [
        item
        for item in bundle.items
        if item.kind
        == UnifiedEvidenceKind.ROBOT_SAMPLE
    ]
    if len(robot_items) != 2:
        raise ValueError(
            "Day 14 reference bundle requires 2 robot items"
        )

    required_refs = [
        *document_item.refs,
        *[
            ref
            for item in robot_items
            for ref in item.refs
        ],
    ]
    required_fact_groups = config[
        "required_fact_groups"
    ]

    healthy_answer = UnifiedGroundedAnswer(
        answer=(
            "6V 7.4V front wrist observation action"
        ),
        abstain=False,
        citations=required_refs,
    )

    scenarios: dict[str, object] = {}

    # 1. Healthy reference over the real Day 12 retrieval bundle.
    scenarios["healthy_reference"] = (
        diagnose_pipeline(
            retrieval_findings=(
                diagnose_retrieval_pages(
                    ranked_pages=[
                        candidate.item.payload.page_number
                        for candidate in document_candidates
                    ],
                    gold_pages=[gold_page],
                    top_k=candidate_pool_k,
                )
            ),
            evidence_findings=diagnose_bundle(
                bundle=bundle,
                required_refs=required_refs,
                require_cross_domain=True,
            ),
            generation_findings=diagnose_generation(
                answer=healthy_answer,
                bundle=bundle,
                required_refs=required_refs,
                required_fact_groups=(
                    required_fact_groups
                ),
                expected_answerable=True,
            ),
        )
    )

    # 2. Retrieval miss: fault-inject the observed ranking by removing
    # the required page. This tests the detector, not retriever quality.
    scenarios["retrieval_miss_injected"] = (
        diagnose_pipeline(
            retrieval_findings=(
                diagnose_retrieval_pages(
                    ranked_pages=[
                        candidate.item.payload.page_number
                        for candidate in document_candidates
                        if (
                            candidate.item.payload.page_number
                            != gold_page
                        )
                    ],
                    gold_pages=[gold_page],
                    top_k=candidate_pool_k,
                )
            )
        )
    )

    # 3. Valid cross-domain bundle but the task-required page is removed.
    missing_page_bundle = UnifiedEvidenceBundle(
        bundle_id="day14_missing_required_page",
        question=bundle.question,
        items=[
            item
            for item in bundle.items
            if not (
                item.kind
                == UnifiedEvidenceKind.DOCUMENT_PAGE
                and item.payload.page_number == gold_page
            )
        ],
    )
    scenarios[
        "missing_required_evidence_injected"
    ] = diagnose_pipeline(
        evidence_findings=diagnose_bundle(
            bundle=missing_page_bundle,
            required_refs=required_refs,
            require_cross_domain=True,
        )
    )

    # 4. Structurally valid bundle with document evidence only.
    docs_only_bundle = UnifiedEvidenceBundle(
        bundle_id="day14_docs_only",
        question=bundle.question,
        items=[
            item
            for item in bundle.items
            if item.kind
            == UnifiedEvidenceKind.DOCUMENT_PAGE
        ],
    )
    scenarios[
        "missing_robot_evidence_injected"
    ] = diagnose_pipeline(
        evidence_findings=diagnose_bundle(
            bundle=docs_only_bundle,
            required_refs=required_refs,
            require_cross_domain=True,
        )
    )

    # 5. Hallucinated citation: the answer cites a page outside bundle.
    hallucinated_ref = EvidenceRef(
        source_id=document_item.provenance.source_id,
        source_type=SourceType.PDF,
        page_number=999,
    )
    hallucinated_answer = UnifiedGroundedAnswer(
        answer=(
            "6V 7.4V front wrist observation action"
        ),
        abstain=False,
        citations=[
            *required_refs,
            hallucinated_ref,
        ],
    )
    scenarios[
        "hallucinated_citation_injected"
    ] = diagnose_pipeline(
        generation_findings=diagnose_generation(
            answer=hallucinated_answer,
            bundle=bundle,
            required_refs=required_refs,
            required_fact_groups=(
                required_fact_groups
            ),
            expected_answerable=True,
        )
    )

    # 6. Citation gap: answer cites only the required PDF page.
    citation_gap_answer = UnifiedGroundedAnswer(
        answer=(
            "6V 7.4V front wrist observation action"
        ),
        abstain=False,
        citations=document_item.refs,
    )
    scenarios[
        "citation_gap_injected"
    ] = diagnose_pipeline(
        generation_findings=diagnose_generation(
            answer=citation_gap_answer,
            bundle=bundle,
            required_refs=required_refs,
            required_fact_groups=(
                required_fact_groups
            ),
            expected_answerable=True,
        )
    )

    # 7. Complete citations but one required fact group is omitted.
    incomplete_answer = UnifiedGroundedAnswer(
        answer=(
            "6V 7.4V front wrist observation"
        ),
        abstain=False,
        citations=required_refs,
    )
    scenarios[
        "incomplete_generation_injected"
    ] = diagnose_pipeline(
        generation_findings=diagnose_generation(
            answer=incomplete_answer,
            bundle=bundle,
            required_refs=required_refs,
            required_fact_groups=(
                required_fact_groups
            ),
            expected_answerable=True,
        )
    )

    # 8. False abstention over an answerable fixture.
    false_abstention = UnifiedGroundedAnswer(
        answer="提供的证据不足以回答该问题",
        abstain=True,
        citations=[],
    )
    scenarios[
        "false_abstention_injected"
    ] = diagnose_pipeline(
        generation_findings=diagnose_generation(
            answer=false_abstention,
            bundle=bundle,
            required_refs=required_refs,
            required_fact_groups=(
                required_fact_groups
            ),
            expected_answerable=True,
        )
    )

    expected = load_expected(
        ROOT / config["scenario_file"]
    )

    rows = []
    all_match = True
    for scenario_id, report in scenarios.items():
        actual_codes = report.codes
        expected_codes = expected[
            scenario_id
        ]
        match = sorted(actual_codes) == sorted(
            expected_codes
        )
        all_match = all_match and match
        rows.append(
            {
                "scenario_id": scenario_id,
                "expected_codes": expected_codes,
                "actual_codes": actual_codes,
                "match": match,
                "healthy": report.healthy,
                "findings": [
                    finding.model_dump(
                        mode="json"
                    )
                    for finding in report.findings
                ],
            }
        )

    result = {
        "mode": (
            "day14_system_failure_diagnosis"
        ),
        "scope": (
            "fault_injection_over_real_evidence_"
            "pipeline_no_robot_outcome_diagnosis"
        ),
        "query": query,
        "episode_id": episode_id,
        "real_reference_bundle_item_count": len(
            bundle.items
        ),
        "real_reference_document_pages": [
            item.payload.page_number
            for item in bundle.items
            if item.kind
            == UnifiedEvidenceKind.DOCUMENT_PAGE
        ],
        "real_reference_robot_frames": [
            item.payload.frame_index
            for item in bundle.items
            if item.kind
            == UnifiedEvidenceKind.ROBOT_SAMPLE
        ],
        "scenario_count": len(rows),
        "all_expected_diagnoses_match": (
            all_match
        ),
        "model_called": False,
        "fault_injection": True,
        "gold_read_by_retrievers": False,
        "robot_event_gold_used": False,
        "real_robot_failure_claimed": False,
        "scenarios": rows,
        "non_claims": [
            "system-pipeline failure diagnosis only",
            "no failed-grasp cause diagnosis",
            "no semantic hallucination detector for uncited prose",
            "hallucination detection is citation-grounding based",
            "no Agent or MCP",
        ],
    }

    output = ROOT / config["report_file"]
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={output}")

    return 0 if all_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
