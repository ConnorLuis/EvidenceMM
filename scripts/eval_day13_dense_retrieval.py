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
            "Day 13 dense retrieval eval has no cases"
        )
    return cases


def metrics_from_ranks(
    ranks: list[int | None],
) -> dict[str, float]:
    n = len(ranks)
    return {
        "hit_at_1": sum(
            rank is not None and rank <= 1
            for rank in ranks
        ) / n,
        "hit_at_3": sum(
            rank is not None and rank <= 3
            for rank in ranks
        ) / n,
        "hit_at_5": sum(
            rank is not None and rank <= 5
            for rank in ranks
        ) / n,
        "mrr_at_5": sum(
            0.0 if rank is None else 1.0 / rank
            for rank in ranks
        ) / n,
    }


def main() -> int:
    config = yaml.safe_load(
        (
            ROOT / "configs/day13_dense_retrieval.yaml"
        ).read_text(encoding="utf-8")
    )

    top_k = int(config["top_k"])
    if top_k != 5:
        raise ValueError(
            "Day 13 dense smoke top_k is frozen to 5"
        )

    documents = load_corpus(
        ROOT / config["text_corpus"]
    )
    if not documents:
        raise ValueError(
            "text corpus must not be empty"
        )

    cases = load_cases(
        ROOT / config["eval_cases"]
    )

    bm25 = BM25Index(documents)

    model, tokenizer, model_load_sec = (
        load_dense_encoder(
            model_name=config["model_name"],
            device=config["device"],
        )
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    document_embeddings, document_encode_sec = (
        encode_dense_texts(
            model=model,
            tokenizer=tokenizer,
            texts=[
                document.text
                for document in documents
            ],
            batch_size=int(config["batch_size"]),
            max_length=int(config["max_length"]),
        )
    )

    rows = []
    bm25_ranks: list[int | None] = []
    dense_ranks: list[int | None] = []
    query_encode_total_sec = 0.0

    started = time.perf_counter()

    for case in cases:
        # Candidate retrieval runs before gold labels are used for metrics.
        bm25_hits = bm25.search(
            case.query,
            top_k=top_k,
        )

        query_embedding, query_encode_sec = (
            encode_dense_texts(
                model=model,
                tokenizer=tokenizer,
                texts=[case.query],
                batch_size=1,
                max_length=int(config["max_length"]),
            )
        )
        query_encode_total_sec += query_encode_sec

        dense_scores = score_dense_documents(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
        )
        dense_hits = rank_dense_scores(
            scores=dense_scores,
            documents=documents,
            top_k=top_k,
        )

        gold = {
            (
                documents[0].source_id,
                page_number,
            )
            for page_number
            in case.document_gold_pages
        }

        bm25_pages = [
            (
                hit.source_id,
                hit.page_number,
            )
            for hit in bm25_hits
        ]
        dense_pages = [
            (
                hit.source_id,
                hit.page_number,
            )
            for hit in dense_hits
        ]

        bm25_rank = first_relevant_rank(
            ranked_pages=bm25_pages,
            gold_pages=gold,
        )
        dense_rank = first_relevant_rank(
            ranked_pages=dense_pages,
            gold_pages=gold,
        )

        bm25_ranks.append(bm25_rank)
        dense_ranks.append(dense_rank)

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
                "bm25_hits": [
                    hit.to_dict()
                    for hit in bm25_hits
                ],
                "dense_hits": [
                    hit.to_dict()
                    for hit in dense_hits
                ],
            }
        )

    evaluation_sec = (
        time.perf_counter() - started
    )

    peak_allocated_gpu_mb = (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )

    bm25_metrics = metrics_from_ranks(
        bm25_ranks
    )
    dense_metrics = metrics_from_ranks(
        dense_ranks
    )

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    after_unload_allocated_gpu_mb = (
        torch.cuda.memory_allocated()
        / 1024
        / 1024
    )

    result = {
        "mode": "day13_dense_retrieval_baseline",
        "scope": (
            "two_case_cross_domain_query_document_"
            "retrieval_smoke_no_hybrid_no_reranker"
        ),
        "model_name": config["model_name"],
        "pooling": "cls",
        "similarity": "cosine",
        "corpus_pages": len(documents),
        "case_count": len(cases),
        "top_k": top_k,
        "gold_read_by_retrievers": False,
        "hybrid_fusion_called": False,
        "reranker_called": False,
        "query_rewrite": False,
        "bm25": bm25_metrics,
        "dense": dense_metrics,
        "delta_dense_minus_bm25": {
            key: (
                dense_metrics[key]
                - bm25_metrics[key]
            )
            for key in bm25_metrics
        },
        "cases": rows,
        "performance": {
            "model_load_sec": model_load_sec,
            "document_encode_sec": (
                document_encode_sec
            ),
            "query_encode_total_sec": (
                query_encode_total_sec
            ),
            "evaluation_sec": evaluation_sec,
            "peak_allocated_gpu_mb": (
                peak_allocated_gpu_mb
            ),
            "after_unload_allocated_gpu_mb": (
                after_unload_allocated_gpu_mb
            ),
        },
        "non_claims": [
            "two-case smoke only",
            "no candidate-recall improvement claim",
            "no sparse+dense hybrid yet",
            "no reranker yet",
            "no ColQwen replacement",
            "no robot retrieval change",
            "no failure diagnosis",
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

    # This gate validates execution, not metric improvement.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
