from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.ranking_analysis import (
    bm25_term_contributions,
    robot_top_tie_summary,
)
from evidencemm.robot_candidate_retrieval import (
    RobotSignalCandidateRetriever,
    rank_robot_signal_samples,
)
from evidencemm.text_retrieval import (
    load_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


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
            / "configs/day13_hybrid_retrieval.yaml"
        ).read_text(encoding="utf-8")
    )
    robot_config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    report_path = ROOT / config[
        "hybrid_report_file"
    ]
    hybrid_report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    case_id = config[
        "analysis_document_case_id"
    ]
    document_case = next(
        (
            case
            for case in hybrid_report[
                "cases"
            ]
            if case["case_id"] == case_id
        ),
        None,
    )
    if document_case is None:
        raise ValueError(
            "analysis document case not found"
        )

    gold_pages = document_case[
        "gold_pages"
    ]
    if len(gold_pages) != 1:
        raise ValueError(
            "ranking analysis expects "
            "one document gold page"
        )
    target_page = int(
        gold_pages[0]
    )

    documents = load_corpus(
        ROOT / config["text_corpus"]
    )
    source_id = (
        document_case[
            "bm25_hits"
        ][0]["source_id"]
    )

    bm25_score, contributions = (
        bm25_term_contributions(
            documents=documents,
            query=document_case["query"],
            source_id=source_id,
            page_number=target_page,
        )
    )

    bm25_hit = next(
        hit
        for hit in document_case[
            "bm25_hits"
        ]
        if hit["page_number"]
        == target_page
    )
    dense_hit = next(
        hit
        for hit in document_case[
            "dense_hits"
        ]
        if hit["page_number"]
        == target_page
    )
    union_hit = next(
        hit
        for hit in document_case[
            "candidate_union"
        ]
        if hit["page_number"]
        == target_page
    )
    reranker_hit = next(
        hit
        for hit in document_case[
            "reranked_hits"
        ]
        if hit["page_number"]
        == target_page
    )

    episode_id = str(
        config["episode_id"]
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

    robot_query = config[
        "analysis_robot_query"
    ]
    robot_profile, all_robot_hits = (
        rank_robot_signal_samples(
            samples=robot_retriever.samples,
            query=robot_query,
            top_k=len(
                robot_retriever.samples
            ),
        )
    )
    robot_summary = (
        robot_top_tie_summary(
            hits=all_robot_hits,
            selected_k=int(
                config[
                    "analysis_robot_selected_k"
                ]
            ),
        )
    )

    result = {
        "mode": (
            "day13_evidence_ranking_analysis"
        ),
        "scope": (
            "document_page_and_robot_signal_"
            "trace_no_failure_diagnosis"
        ),
        "document": {
            "case_id": case_id,
            "query": document_case[
                "query"
            ],
            "target_page": target_page,
            "bm25": {
                "rank": bm25_hit[
                    "rank"
                ],
                "reported_score": (
                    bm25_hit["score"]
                ),
                "recomputed_score": (
                    bm25_score
                ),
                "top_term_contributions": [
                    row.to_dict()
                    for row
                    in contributions[:12]
                ],
                "explanation": (
                    "BM25 rank follows exact "
                    "lexical term contributions "
                    "under the frozen BM25 formula."
                ),
            },
            "dense": {
                "rank": dense_hit[
                    "rank"
                ],
                "cosine_similarity": (
                    dense_hit["score"]
                ),
                "explanation": (
                    "Dense branch uses BGE-M3 "
                    "CLS embedding cosine similarity; "
                    "no token-level attribution is claimed."
                ),
            },
            "candidate_union": {
                **union_hit,
                "explanation": (
                    "The page enters the candidate "
                    "pool if retrieved by either branch; "
                    "pool_index is not a relevance rank."
                ),
            },
            "reranker": {
                "rank": reranker_hit[
                    "rank"
                ],
                "raw_relevance_logit": (
                    reranker_hit[
                        "reranker_score"
                    ]
                ),
                "explanation": (
                    "Cross-encoder score is a raw "
                    "relevance logit, not a probability."
                ),
            },
        },
        "robot": {
            "query": robot_query,
            "profile": (
                robot_profile.to_dict()
            ),
            **robot_summary,
            "explanation": (
                "For canonical query 'robot gripper action', "
                "the baseline scores only gripper action "
                "change. Equal top scores are ordered by "
                "lower frame_index, so the first two frames "
                "under the frozen robot quota are selected."
            ),
        },
        "non_claims": [
            "ranking trace, not causal diagnosis",
            "dense similarity is not token attribution",
            "reranker logit is not calibrated probability",
            "robot signal ranking is not event semantics",
            "no failure diagnosis",
            "no Agent or MCP",
        ],
    }

    output = ROOT / config[
        "ranking_report_file"
    ]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
