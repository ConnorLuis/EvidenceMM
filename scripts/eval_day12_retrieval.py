from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.document_candidate_retrieval import (
    DocumentBM25CandidateRetriever,
)
from evidencemm.retrieval import validate_retrieved_bundle
from evidencemm.retrieval_eval import (
    Day12RetrievalEvalCase,
    evaluate_document_candidates,
    robot_profile_matches,
    validate_robot_candidate_evidence,
)
from evidencemm.retrieval_ranking import (
    DAY12_BASELINE_BUDGET,
    compose_fixed_quota,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalCandidateRetriever,
)
from evidencemm.unified_evidence import UnifiedEvidenceKind


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = (
    ROOT / "data/eval/day12_retrieval_cases.jsonl"
)


def load_cases() -> list[Day12RetrievalEvalCase]:
    cases = [
        Day12RetrievalEvalCase.model_validate_json(line)
        for line in CASES_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("Day 12 retrieval eval has no cases")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-dir",
        required=True,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT / "configs/day12_retrieval.yaml"
        ).read_text(encoding="utf-8")
    )
    robot_config = yaml.safe_load(
        (
            ROOT / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    candidate_pool_k = int(
        config["candidate_pool_k"]
    )
    if candidate_pool_k != 5:
        raise ValueError(
            "Day 12 candidate_pool_k must remain 5"
        )

    document_retriever = DocumentBM25CandidateRetriever(
        project_root=ROOT,
        source_manifest_path=config["document_manifest"],
        visual_manifest_path=config["document_visual_manifest"],
    )

    episode_id = str(config["episode_id"])
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

    robot_retriever = RobotSignalCandidateRetriever(
        project_root=ROOT,
        episode_manifest_path=episode_manifest_path,
        episode_dir=args.episode_dir,
        frame_records_path=frame_records_path,
    )

    results = []
    all_structural_valid = True

    for case in load_cases():
        document_candidates = document_retriever.search(
            case.query,
            top_k=candidate_pool_k,
        )
        robot_candidates = robot_retriever.search(
            case.query,
            top_k=candidate_pool_k,
        )

        composition = compose_fixed_quota(
            query=case.query,
            document_candidates=document_candidates,
            robot_candidates=robot_candidates,
            budget=DAY12_BASELINE_BUDGET,
            bundle_id=f"day12_eval_{case.case_id}",
        )

        validate_retrieved_bundle(
            query=case.query,
            top_k=DAY12_BASELINE_BUDGET.total_top_k,
            bundle=composition.bundle,
        )

        document_metrics = evaluate_document_candidates(
            candidates=document_candidates,
            gold_pages=case.document_gold_pages,
            candidate_pool_k=candidate_pool_k,
        )

        profile = robot_retriever.query_profile(
            case.query
        )
        profile_valid = robot_profile_matches(
            profile=profile,
            expected_joints=case.expected_robot_joints,
            expected_signals=case.expected_robot_signals,
        )

        robot_evidence_valid, robot_errors = (
            validate_robot_candidate_evidence(
                robot_candidates
            )
        )

        selected_document_count = sum(
            item.kind
            == UnifiedEvidenceKind.DOCUMENT_PAGE
            for item in composition.bundle.items
        )
        selected_robot_count = sum(
            item.kind
            == UnifiedEvidenceKind.ROBOT_SAMPLE
            for item in composition.bundle.items
        )

        structural_valid = (
            profile_valid
            and robot_evidence_valid
            and len(composition.bundle.items) == 5
            and selected_document_count == 3
            and selected_robot_count == 2
        )
        all_structural_valid = (
            all_structural_valid
            and structural_valid
        )

        results.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "document": (
                    document_metrics.model_dump(
                        mode="json"
                    )
                ),
                "robot_profile": profile.to_dict(),
                "robot_profile_valid": profile_valid,
                "robot_candidate_evidence_valid": (
                    robot_evidence_valid
                ),
                "robot_candidate_evidence_errors": (
                    robot_errors
                ),
                "robot_top_frames": [
                    {
                        "rank": candidate.rank,
                        "frame_index": (
                            candidate.item.payload.frame_index
                        ),
                        "timestamp_sec": (
                            candidate.item.payload.timestamp_sec
                        ),
                        "raw_score": candidate.raw_score,
                    }
                    for candidate in robot_candidates
                ],
                "bundle_item_count": len(
                    composition.bundle.items
                ),
                "selected_document_count": (
                    selected_document_count
                ),
                "selected_robot_count": (
                    selected_robot_count
                ),
                "structural_valid": structural_valid,
            }
        )

    document_hit_at_1 = (
        sum(
            int(result["document"]["hit_at_1"])
            for result in results
        )
        / len(results)
    )
    document_hit_at_3 = (
        sum(
            int(result["document"]["hit_at_3"])
            for result in results
        )
        / len(results)
    )
    document_hit_at_5 = (
        sum(
            int(result["document"]["hit_at_5"])
            for result in results
        )
        / len(results)
    )
    document_mrr = (
        sum(
            float(
                result["document"]["reciprocal_rank"]
            )
            for result in results
        )
        / len(results)
    )
    robot_profile_accuracy = (
        sum(
            int(result["robot_profile_valid"])
            for result in results
        )
        / len(results)
    )
    bundle_valid_rate = (
        sum(
            int(result["structural_valid"])
            for result in results
        )
        / len(results)
    )

    payload = {
        "mode": "day12_retrieval_evaluation",
        "scope": (
            "two_case_smoke_eval_no_generation_"
            "no_event_gold_no_failure_diagnosis"
        ),
        "case_count": len(results),
        "candidate_pool_k_per_domain": (
            candidate_pool_k
        ),
        "budget": {
            "total_top_k": (
                DAY12_BASELINE_BUDGET.total_top_k
            ),
            "document_quota": (
                DAY12_BASELINE_BUDGET.document_quota
            ),
            "robot_quota": (
                DAY12_BASELINE_BUDGET.robot_quota
            ),
        },
        "evaluation_labels_loaded": True,
        "gold_read_by_retriever": False,
        "robot_event_gold_used": False,
        "generation_called": False,
        "failure_diagnosis_attempted": False,
        "agent_used": False,
        "cross_domain_score_fusion": False,
        "query_rewrite": False,
        "results": results,
        "aggregate": {
            "document_hit_at_1": document_hit_at_1,
            "document_hit_at_3": document_hit_at_3,
            "document_hit_at_5": document_hit_at_5,
            "document_mrr": document_mrr,
            "robot_profile_accuracy": (
                robot_profile_accuracy
            ),
            "bundle_structural_valid_rate": (
                bundle_valid_rate
            ),
        },
        "evaluation_completed": True,
        "non_claims": [
            "two-case smoke evaluation only",
            "no benchmark-scale retrieval claim",
            "no robot event-recognition metric",
            "no failure diagnosis",
            "no grounded generation",
        ],
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if all_structural_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
