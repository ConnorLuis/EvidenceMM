from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.document_candidate_retrieval import (
    DocumentBM25CandidateRetriever,
)
from evidencemm.retrieval import validate_retrieved_bundle
from evidencemm.retrieval_ranking import (
    DAY12_BASELINE_BUDGET,
    compose_fixed_quota,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalCandidateRetriever,
)
from evidencemm.unified_evidence import UnifiedEvidenceKind


ROOT = Path(__file__).resolve().parents[1]


def candidate_trace(candidate) -> dict:
    item = candidate.item
    payload = item.payload

    row = {
        "domain": candidate.domain.value,
        "retriever_name": candidate.retriever_name,
        "rank": candidate.rank,
        "raw_score": candidate.raw_score,
        "evidence_id": item.evidence_id,
        "kind": item.kind.value,
    }

    if item.kind == UnifiedEvidenceKind.DOCUMENT_PAGE:
        row["page_number"] = payload.page_number
        row["source_id"] = item.provenance.source_id
    else:
        row["episode_id"] = payload.episode_id
        row["frame_index"] = payload.frame_index
        row["timestamp_sec"] = payload.timestamp_sec
        row["cameras"] = [
            camera.camera
            for camera in payload.cameras
        ]

    return row


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
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    query = str(config["query"]).strip()
    candidate_pool_k = int(
        config["candidate_pool_k"]
    )
    if candidate_pool_k != 5:
        raise ValueError(
            "Day 12 candidate_pool_k is frozen to 5"
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

    document_candidates = document_retriever.search(
        query,
        top_k=candidate_pool_k,
    )
    robot_candidates = robot_retriever.search(
        query,
        top_k=candidate_pool_k,
    )

    composition = compose_fixed_quota(
        query=query,
        document_candidates=document_candidates,
        robot_candidates=robot_candidates,
        budget=DAY12_BASELINE_BUDGET,
        bundle_id=f"day12_retrieval_{episode_id}",
    )

    validate_retrieved_bundle(
        query=query,
        top_k=DAY12_BASELINE_BUDGET.total_top_k,
        bundle=composition.bundle,
    )

    selected_document_count = sum(
        item.kind == UnifiedEvidenceKind.DOCUMENT_PAGE
        for item in composition.bundle.items
    )
    selected_robot_count = sum(
        item.kind == UnifiedEvidenceKind.ROBOT_SAMPLE
        for item in composition.bundle.items
    )

    payload = {
        "mode": "day12_real_evidence_search_smoke",
        "scope": "retrieval_only_no_generation_no_gold",
        "query": query,
        "gold_read": False,
        "generation_called": False,
        "failure_diagnosis_attempted": False,
        "agent_used": False,
        "candidate_pool_k_per_domain": candidate_pool_k,
        "budget": {
            "total_top_k": DAY12_BASELINE_BUDGET.total_top_k,
            "document_quota": DAY12_BASELINE_BUDGET.document_quota,
            "robot_quota": DAY12_BASELINE_BUDGET.robot_quota,
        },
        "robot_query_profile": (
            robot_retriever.query_profile(query).to_dict()
        ),
        "document_candidates": [
            candidate_trace(candidate)
            for candidate in document_candidates
        ],
        "robot_candidates": [
            candidate_trace(candidate)
            for candidate in robot_candidates
        ],
        "selected_candidates": [
            candidate_trace(candidate)
            for candidate in composition.selected_candidates
        ],
        "bundle": {
            "schema_version": composition.bundle.schema_version,
            "bundle_id": composition.bundle.bundle_id,
            "question": composition.bundle.question,
            "item_count": len(composition.bundle.items),
            "document_item_count": selected_document_count,
            "robot_item_count": selected_robot_count,
        },
        "cross_domain_score_fusion": False,
        "query_rewrite": False,
        "next_connection_not_claimed": (
            "grounded generation is not called "
            "by this retrieval-only smoke"
        ),
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    expected = (
        len(composition.bundle.items) == 5
        and selected_document_count == 3
        and selected_robot_count == 2
    )
    return 0 if expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
