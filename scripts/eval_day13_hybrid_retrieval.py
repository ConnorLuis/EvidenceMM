from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import torch
import yaml

from evidencemm.dense_retrieval import (
    encode_dense_texts,
    first_relevant_rank,
    load_dense_encoder,
    rank_dense_scores,
    score_dense_documents,
)
from evidencemm.hybrid_candidate_union import (
    branch_overlap_count,
    build_candidate_union,
    candidate_pool_contains_gold,
)
from evidencemm.reranking import (
    load_reranker,
    rank_reranker_scores,
    score_query_passages,
)
from evidencemm.retrieval_eval import (
    Day12RetrievalEvalCase,
)
from evidencemm.text_retrieval import (
    BM25Index,
    load_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


def load_cases(
    path: Path,
) -> list[Day12RetrievalEvalCase]:
    cases = [
        Day12RetrievalEvalCase.model_validate_json(
            line
        )
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(
            "Day 13 hybrid eval has no cases"
        )
    return cases


def ranking_metrics(
    ranks: list[int | None],
) -> dict[str, float]:
    n = len(ranks)
    return {
        "hit_at_1": sum(
            rank is not None
            and rank <= 1
            for rank in ranks
        )
        / n,
        "hit_at_3": sum(
            rank is not None
            and rank <= 3
            for rank in ranks
        )
        / n,
        "hit_at_5": sum(
            rank is not None
            and rank <= 5
            for rank in ranks
        )
        / n,
        "mrr_at_5": sum(
            0.0
            if rank is None
            else 1.0 / rank
            for rank in ranks
        )
        / n,
    }


def main() -> int:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/day13_hybrid_retrieval.yaml"
        ).read_text(encoding="utf-8")
    )

    branch_top_k = int(
        config["branch_top_k"]
    )
    final_top_k = int(
        config["final_top_k"]
    )
    if (
        branch_top_k != 5
        or final_top_k != 5
    ):
        raise ValueError(
            "Day 13 branch/final top_k "
            "must remain 5"
        )

    documents = load_corpus(
        ROOT / config["text_corpus"]
    )
    document_by_key = {
        (
            document.source_id,
            document.page_number,
        ): document
        for document in documents
    }
    cases = load_cases(
        ROOT / config["eval_cases"]
    )

    bm25 = BM25Index(documents)

    # Phase 1: BM25 + dense retrieval and candidate union.
    dense_model, dense_tokenizer, (
        dense_model_load_sec
    ) = load_dense_encoder(
        model_name=config[
            "dense_model_name"
        ],
        device=config["device"],
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    document_embeddings, (
        document_encode_sec
    ) = encode_dense_texts(
        model=dense_model,
        tokenizer=dense_tokenizer,
        texts=[
            document.text
            for document in documents
        ],
        batch_size=int(
            config["dense_batch_size"]
        ),
        max_length=int(
            config["dense_max_length"]
        ),
    )

    pending = []
    dense_query_encode_sec = 0.0

    for case in cases:
        bm25_hits = bm25.search(
            case.query,
            top_k=branch_top_k,
        )

        query_embedding, elapsed = (
            encode_dense_texts(
                model=dense_model,
                tokenizer=dense_tokenizer,
                texts=[case.query],
                batch_size=1,
                max_length=int(
                    config[
                        "dense_max_length"
                    ]
                ),
            )
        )
        dense_query_encode_sec += elapsed

        dense_scores = score_dense_documents(
            query_embedding=query_embedding,
            document_embeddings=(
                document_embeddings
            ),
        )
        dense_hits = rank_dense_scores(
            scores=dense_scores,
            documents=documents,
            top_k=branch_top_k,
        )

        pool = build_candidate_union(
            bm25_hits=bm25_hits,
            dense_hits=dense_hits,
        )

        # Gold is deliberately not touched before retrieval + union.
        pending.append(
            {
                "case": case,
                "bm25_hits": bm25_hits,
                "dense_hits": dense_hits,
                "pool": pool,
            }
        )

    dense_peak_gpu_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    del dense_model
    del dense_tokenizer
    del document_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: cross-encoder reranking over the union only.
    reranker, reranker_tokenizer, (
        reranker_model_load_sec
    ) = load_reranker(
        model_name=config[
            "reranker_model_name"
        ],
        device=config["device"],
    )

    torch.cuda.reset_peak_memory_stats()

    reranker_total_sec = 0.0
    for row in pending:
        pool = row["pool"]
        passages = [
            document_by_key[
                (
                    candidate.source_id,
                    candidate.page_number,
                )
            ].text
            for candidate in pool
        ]
        scores, elapsed = (
            score_query_passages(
                model=reranker,
                tokenizer=reranker_tokenizer,
                query=row["case"].query,
                passages=passages,
                batch_size=int(
                    config[
                        "reranker_batch_size"
                    ]
                ),
                max_length=int(
                    config[
                        "reranker_max_length"
                    ]
                ),
            )
        )
        reranker_total_sec += elapsed
        row["reranked_hits"] = (
            rank_reranker_scores(
                pool=pool,
                scores=scores,
                top_k=final_top_k,
            )
        )

    reranker_peak_gpu_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    del reranker
    del reranker_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    after_unload_allocated_gpu_mb = (
        torch.cuda.memory_allocated()
        / 1024
        / 1024
    )

    # Phase 3: evaluation only after all retrieval/reranking outputs exist.
    bm25_ranks: list[int | None] = []
    dense_ranks: list[int | None] = []
    reranker_ranks: list[
        int | None
    ] = []
    union_hits = 0
    rows = []

    for row in pending:
        case = row["case"]
        bm25_hits = row["bm25_hits"]
        dense_hits = row["dense_hits"]
        pool = row["pool"]
        reranked_hits = row[
            "reranked_hits"
        ]

        source_id = (
            bm25_hits[0].source_id
            if bm25_hits
            else dense_hits[0].source_id
        )
        gold = {
            (
                source_id,
                page_number,
            )
            for page_number
            in case.document_gold_pages
        }

        bm25_rank = first_relevant_rank(
            ranked_pages=[
                (
                    hit.source_id,
                    hit.page_number,
                )
                for hit in bm25_hits
            ],
            gold_pages=gold,
        )
        dense_rank = first_relevant_rank(
            ranked_pages=[
                (
                    hit.source_id,
                    hit.page_number,
                )
                for hit in dense_hits
            ],
            gold_pages=gold,
        )
        reranker_rank = (
            first_relevant_rank(
                ranked_pages=[
                    (
                        hit.source_id,
                        hit.page_number,
                    )
                    for hit in reranked_hits
                ],
                gold_pages=gold,
            )
        )
        union_contains = (
            candidate_pool_contains_gold(
                pool=pool,
                gold_pages=gold,
            )
        )

        bm25_ranks.append(bm25_rank)
        dense_ranks.append(dense_rank)
        reranker_ranks.append(
            reranker_rank
        )
        union_hits += int(
            union_contains
        )

        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "gold_pages": (
                    case.document_gold_pages
                ),
                "bm25_first_relevant_rank": (
                    bm25_rank
                ),
                "dense_first_relevant_rank": (
                    dense_rank
                ),
                "union_contains_gold": (
                    union_contains
                ),
                "reranker_first_relevant_rank": (
                    reranker_rank
                ),
                "bm25_hits": [
                    hit.to_dict()
                    for hit in bm25_hits
                ],
                "dense_hits": [
                    hit.to_dict()
                    for hit in dense_hits
                ],
                "candidate_union": [
                    candidate.to_dict()
                    for candidate in pool
                ],
                "candidate_union_size": len(
                    pool
                ),
                "branch_overlap_count": (
                    branch_overlap_count(
                        pool
                    )
                ),
                "reranked_hits": [
                    hit.to_dict()
                    for hit in reranked_hits
                ],
            }
        )

    bm25_metrics = ranking_metrics(
        bm25_ranks
    )
    dense_metrics = ranking_metrics(
        dense_ranks
    )
    reranker_metrics = ranking_metrics(
        reranker_ranks
    )

    n = len(rows)
    union_candidate_recall = (
        union_hits / n
    )
    bm25_candidate_recall_at_5 = (
        bm25_metrics["hit_at_5"]
    )

    result = {
        "mode": (
            "day13_sparse_dense_"
            "reranker_evaluation"
        ),
        "scope": (
            "two_case_document_retrieval_"
            "smoke_with_union_and_cross_encoder"
        ),
        "dense_model_name": config[
            "dense_model_name"
        ],
        "reranker_model_name": config[
            "reranker_model_name"
        ],
        "reranker_score_semantics": (
            "raw relevance logit; "
            "not calibrated probability"
        ),
        "corpus_pages": len(documents),
        "case_count": n,
        "branch_top_k": branch_top_k,
        "final_top_k": final_top_k,
        "gold_read_by_retrievers": False,
        "raw_score_fusion": False,
        "union_order_is_relevance_rank": (
            False
        ),
        "query_rewrite": False,
        "bm25": bm25_metrics,
        "dense": dense_metrics,
        "candidate_union": {
            "candidate_recall": (
                union_candidate_recall
            ),
            "average_pool_size": sum(
                row[
                    "candidate_union_size"
                ]
                for row in rows
            )
            / n,
            "average_branch_overlap": sum(
                row[
                    "branch_overlap_count"
                ]
                for row in rows
            )
            / n,
            "bm25_hit_at_5_reference": (
                bm25_candidate_recall_at_5
            ),
            "coverage_gain_over_bm25_top5": (
                union_candidate_recall
                - bm25_candidate_recall_at_5
            ),
            "same_budget_recall_claim": (
                False
            ),
        },
        "reranker": reranker_metrics,
        "delta_reranker_minus_bm25": {
            key: (
                reranker_metrics[key]
                - bm25_metrics[key]
            )
            for key in bm25_metrics
        },
        "cases": rows,
        "performance": {
            "dense_model_load_sec": (
                dense_model_load_sec
            ),
            "dense_document_encode_sec": (
                document_encode_sec
            ),
            "dense_query_encode_total_sec": (
                dense_query_encode_sec
            ),
            "dense_peak_allocated_gpu_mb": (
                dense_peak_gpu_mb
            ),
            "reranker_model_load_sec": (
                reranker_model_load_sec
            ),
            "reranker_total_sec": (
                reranker_total_sec
            ),
            "reranker_peak_allocated_gpu_mb": (
                reranker_peak_gpu_mb
            ),
            "after_unload_allocated_gpu_mb": (
                after_unload_allocated_gpu_mb
            ),
        },
        "non_claims": [
            "two-case smoke only",
            "no benchmark-scale recall claim",
            "union uses a larger variable candidate pool",
            "no direct BM25+dense raw-score addition",
            "no ColQwen replacement",
            "no robot retrieval modification",
            "no failure diagnosis",
            "no Agent or MCP",
        ],
    }

    output = ROOT / config[
        "hybrid_report_file"
    ]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
