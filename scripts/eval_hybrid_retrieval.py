from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import yaml

from evidencemm.hybrid_retrieval import RankedPage, fuse_rrf
from evidencemm.schemas import EvalCase, SourceType
from evidencemm.text_retrieval import BM25Index, load_corpus
from evidencemm.visual_retrieval import (
    encode_query,
    load_model_and_processor,
    rank_scores,
    score_documents,
)


ROOT = Path(__file__).resolve().parents[1]


def load_verified_pdf_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = EvalCase.model_validate(json.loads(line))
        if any(
            evidence.source_type == SourceType.PDF
            for evidence in case.expected_evidence
        ):
            cases.append(case)
    return cases


def dcg_at_k(
    ranked: list[tuple[str, int]],
    gold: set[tuple[str, int]],
    k: int,
) -> float:
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in gold
    )


def main() -> int:
    config = yaml.safe_load(
        (ROOT / "configs/hybrid_retrieval.yaml").read_text(
            encoding="utf-8"
        )
    )
    text_config = yaml.safe_load(
        (ROOT / config["text_config"]).read_text(
            encoding="utf-8"
        )
    )
    visual_config = yaml.safe_load(
        (ROOT / config["visual_config"]).read_text(
            encoding="utf-8"
        )
    )
    top_k = int(config["top_k"])

    text_documents = load_corpus(ROOT / config["text_corpus"])
    text_index = BM25Index(
        text_documents,
        k1=float(text_config["k1"]),
        b=float(text_config["b"]),
    )

    visual_index = torch.load(
        ROOT / config["visual_index"],
        map_location="cpu",
        weights_only=False,
    )
    model, processor, model_load_sec = load_model_and_processor(
        model_name=visual_config["model_name"],
        max_num_visual_tokens=int(
            visual_config["max_num_visual_tokens"]
        ),
    )

    cases = load_verified_pdf_cases(
        ROOT / "data/eval/day2_verified_cases.jsonl"
    )
    if len(cases) != 2:
        raise SystemExit(
            f"expected 2 verified PDF cases, got {len(cases)}"
        )

    recall_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    rows = []

    started = time.perf_counter()

    for case in cases:
        gold = {
            (evidence.source_id, evidence.page_number)
            for evidence in case.expected_evidence
            if (
                evidence.source_type == SourceType.PDF
                and evidence.page_number is not None
            )
        }

        text_raw_hits = text_index.search(
            case.question,
            top_k=top_k,
        )
        text_hits = [
            RankedPage(
                source_id=hit.source_id,
                page_number=hit.page_number,
                rank=hit.rank,
                raw_score=hit.score,
            )
            for hit in text_raw_hits
        ]

        query_embedding, query_encode_sec = encode_query(
            model=model,
            processor=processor,
            query=case.question,
        )
        visual_scores = score_documents(
            processor=processor,
            query_embedding=query_embedding,
            document_embeddings=visual_index["embeddings"],
            device=model.device,
        )
        visual_ranking = rank_scores(
            visual_scores,
            top_k=top_k,
        )

        vision_hits = []
        for rank, (doc_index, score) in enumerate(
            visual_ranking,
            start=1,
        ):
            page = visual_index["pages"][doc_index]
            vision_hits.append(
                RankedPage(
                    source_id=page["source_id"],
                    page_number=page["page_number"],
                    rank=rank,
                    raw_score=score,
                )
            )

        hybrid_hits = fuse_rrf(
            text_hits=text_hits,
            vision_hits=vision_hits,
            top_k=top_k,
            rrf_k=int(config["rrf_k"]),
            text_weight=float(config["text_weight"]),
            vision_weight=float(config["vision_weight"]),
        )

        ranked_items = [
            (hit.source_id, hit.page_number)
            for hit in hybrid_hits
        ]

        for cutoff in [1, 3, 5]:
            if any(
                item in gold
                for item in ranked_items[:cutoff]
            ):
                recall_hits[cutoff] += 1

        first_rank = next(
            (
                rank
                for rank, item in enumerate(
                    ranked_items,
                    start=1,
                )
                if item in gold
            ),
            None,
        )
        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank

        ideal_relevant = min(len(gold), top_k)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant + 1)
        )
        ndcg_sum += (
            dcg_at_k(ranked_items, gold, top_k) / ideal_dcg
            if ideal_dcg
            else 0.0
        )

        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "gold_pages": [
                    {
                        "source_id": source_id,
                        "page_number": page_number,
                    }
                    for source_id, page_number in sorted(gold)
                ],
                "first_relevant_rank": first_rank,
                "query_encode_sec": query_encode_sec,
                "text_hits": [hit.__dict__ for hit in text_hits],
                "vision_hits": [hit.__dict__ for hit in vision_hits],
                "hybrid_hits": [
                    hit.to_dict() for hit in hybrid_hits
                ],
            }
        )

    torch.cuda.synchronize()
    evaluation_sec = time.perf_counter() - started

    n = len(cases)
    result = {
        "mode": "hybrid_rrf",
        "corpus_pages": len(text_documents),
        "verified_pdf_cases": n,
        "rrf_k": int(config["rrf_k"]),
        "text_weight": float(config["text_weight"]),
        "vision_weight": float(config["vision_weight"]),
        "recall_at_1": recall_hits[1] / n,
        "recall_at_3": recall_hits[3] / n,
        "recall_at_5": recall_hits[5] / n,
        "mrr_at_5": reciprocal_rank_sum / n,
        "ndcg_at_5": ndcg_sum / n,
        "model_load_sec": model_load_sec,
        "evaluation_sec": evaluation_sec,
        "cases": rows,
        "scope_note": (
            "Day 5 smoke hybrid baseline over one 8-page "
            "datasheet and two verified PDF queries; "
            "not a headline benchmark."
        ),
    }

    for label, filename in [
        ("day3_text_only_reference", "day3_text_retrieval.json"),
        ("day4_vision_only_reference", "day4_visual_retrieval.json"),
    ]:
        path = ROOT / "reports" / filename
        if path.exists():
            previous = json.loads(
                path.read_text(encoding="utf-8")
            )
            result[label] = {
                key: previous[key]
                for key in [
                    "recall_at_1",
                    "recall_at_3",
                    "recall_at_5",
                    "mrr_at_5",
                    "ndcg_at_5",
                ]
            }

    output = ROOT / config["report_file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
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
