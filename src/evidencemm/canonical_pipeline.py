from __future__ import annotations

from pathlib import Path

import yaml

from evidencemm.canonical_document_retriever import (
    CanonicalHybridDocumentRetriever,
)
from evidencemm.document_candidate_retrieval import (
    DocumentBM25CandidateRetriever,
)
from evidencemm.retrieval import validate_retrieved_bundle
from evidencemm.retrieval_ranking import (
    DAY12_BASELINE_BUDGET,
    RetrievalComposition,
    compose_fixed_quota,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalCandidateRetriever,
)
from evidencemm.unified_evidence import (
    validate_cross_domain_bundle,
)


VALID_DOCUMENT_MODES = {
    "bm25",
    "hybrid",
}


def validate_document_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in VALID_DOCUMENT_MODES:
        raise ValueError(
            "document_mode must be one of: "
            + ", ".join(
                sorted(VALID_DOCUMENT_MODES)
            )
        )
    return normalized


def build_canonical_retrieval(
    *,
    project_root: str | Path,
    episode_dir: str | Path,
    query: str,
    document_mode: str,
    generation_config_path: str | Path = (
        "configs/day12_retrieval_grounded_generation.yaml"
    ),
    hybrid_config_path: str | Path = (
        "configs/day13_hybrid_retrieval.yaml"
    ),
    robot_config_path: str | Path = (
        "configs/robot_sequence_evidence.yaml"
    ),
) -> tuple[
    RetrievalComposition,
    dict,
]:
    root = Path(project_root).resolve()
    mode = validate_document_mode(
        document_mode
    )

    generation_config_file = Path(
        generation_config_path
    )
    if not generation_config_file.is_absolute():
        generation_config_file = (
            root / generation_config_file
        )
    config = yaml.safe_load(
        generation_config_file.read_text(
            encoding="utf-8"
        )
    )

    robot_config_file = Path(
        robot_config_path
    )
    if not robot_config_file.is_absolute():
        robot_config_file = (
            root / robot_config_file
        )
    robot_config = yaml.safe_load(
        robot_config_file.read_text(
            encoding="utf-8"
        )
    )

    candidate_pool_k = int(
        config["candidate_pool_k"]
    )
    if candidate_pool_k != 5:
        raise ValueError(
            "canonical Day15 candidate_pool_k must remain 5"
        )

    if mode == "bm25":
        document_retriever = (
            DocumentBM25CandidateRetriever(
                project_root=root,
                source_manifest_path=(
                    config["document_manifest"]
                ),
                visual_manifest_path=(
                    config[
                        "document_visual_manifest"
                    ]
                ),
            )
        )
    else:
        document_retriever = (
            CanonicalHybridDocumentRetriever(
                project_root=root,
                source_manifest_path=(
                    config["document_manifest"]
                ),
                visual_manifest_path=(
                    config[
                        "document_visual_manifest"
                    ]
                ),
                hybrid_config_path=(
                    hybrid_config_path
                ),
            )
        )

    episode_id = str(
        config["episode_id"]
    )
    episode_manifest_path = (
        root
        / robot_config["manifest_root"]
        / f"{episode_id}.json"
    )
    frame_records_path = (
        root
        / robot_config["processed_root"]
        / episode_id
        / "frames.jsonl"
    )

    robot_retriever = (
        RobotSignalCandidateRetriever(
            project_root=root,
            episode_manifest_path=(
                episode_manifest_path
            ),
            episode_dir=episode_dir,
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
    robot_candidates = robot_retriever.search(
        query,
        top_k=candidate_pool_k,
    )

    if hasattr(
        document_retriever,
        "release_models",
    ):
        document_retriever.release_models()

    composition = compose_fixed_quota(
        query=query,
        document_candidates=document_candidates,
        robot_candidates=robot_candidates,
        budget=DAY12_BASELINE_BUDGET,
        bundle_id=(
            f"day15_canonical_{mode}_{episode_id}"
        ),
    )

    validate_retrieved_bundle(
        query=query,
        top_k=(
            DAY12_BASELINE_BUDGET.total_top_k
        ),
        bundle=composition.bundle,
    )

    valid, errors = (
        validate_cross_domain_bundle(
            composition.bundle
        )
    )
    if not valid:
        raise ValueError(
            "invalid canonical cross-domain bundle: "
            + repr(errors)
        )

    trace = {
        "document_mode": mode,
        "document_retriever_name": (
            document_candidates[
                0
            ].retriever_name
        ),
        "document_candidates": [
            {
                "rank": candidate.rank,
                "raw_score": (
                    candidate.raw_score
                ),
                "page_number": (
                    candidate.item.payload.page_number
                ),
            }
            for candidate in document_candidates
        ],
        "robot_retriever_name": (
            robot_candidates[0].retriever_name
        ),
        "robot_candidates": [
            {
                "rank": candidate.rank,
                "raw_score": (
                    candidate.raw_score
                ),
                "frame_index": (
                    candidate.item.payload.frame_index
                ),
                "timestamp_sec": (
                    candidate.item.payload.timestamp_sec
                ),
            }
            for candidate in robot_candidates
        ],
        "document_ranking_trace": (
            document_retriever.last_trace.to_dict()
            if (
                mode == "hybrid"
                and document_retriever.last_trace
                is not None
            )
            else None
        ),
    }

    return composition, trace
