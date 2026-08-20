from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from evidencemm.schemas import (
    EvalCase,
    SourceType,
)
from evidencemm.text_retrieval import (
    BM25Index,
    load_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Day 3 text-only BM25 baseline "
            "against verified PDF evidence pages."
        )
    )
    parser.add_argument(
        "--corpus",
        default=(
            "data/processed/text/"
            "sts3215_pages.jsonl"
        ),
    )
    parser.add_argument(
        "--verified",
        default=(
            "data/eval/"
            "day2_verified_cases.jsonl"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/text_retrieval.yaml",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "day3_text_retrieval.json"
        ),
    )
    return parser.parse_args()


def load_verified_pdf_cases(
    path: Path,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        case = EvalCase.model_validate(
            json.loads(line)
        )
        if any(
            evidence.source_type
            == SourceType.PDF
            for evidence
            in case.expected_evidence
        ):
            cases.append(case)

    return cases


def dcg_at_k(
    ranked: list[tuple[str, int]],
    gold: set[tuple[str, int]],
    k: int,
) -> float:
    score = 0.0
    for rank, item in enumerate(
        ranked[:k],
        start=1,
    ):
        if item in gold:
            score += 1.0 / math.log2(rank + 1)
    return score


def main() -> int:
    args = parse_args()

    config = yaml.safe_load(
        (ROOT / args.config).read_text(
            encoding="utf-8"
        )
    )
    top_k = int(config["top_k"])

    documents = load_corpus(
        ROOT / args.corpus
    )
    index = BM25Index(
        documents,
        k1=float(config["k1"]),
        b=float(config["b"]),
    )
    cases = load_verified_pdf_cases(
        ROOT / args.verified
    )

    if not cases:
        raise SystemExit(
            "no verified PDF cases found"
        )

    cutoffs = [1, 3, 5]
    recall_hits = {
        k: 0 for k in cutoffs
    }
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    case_rows: list[dict] = []

    for case in cases:
        gold = {
            (
                evidence.source_id,
                evidence.page_number,
            )
            for evidence
            in case.expected_evidence
            if (
                evidence.source_type
                == SourceType.PDF
                and evidence.page_number
                is not None
            )
        }

        hits = index.search(
            case.question,
            top_k=top_k,
        )
        ranked = [
            (hit.source_id, hit.page_number)
            for hit in hits
        ]

        for k in cutoffs:
            if any(
                item in gold
                for item in ranked[:k]
            ):
                recall_hits[k] += 1

        first_relevant_rank = next(
            (
                rank
                for rank, item
                in enumerate(
                    ranked,
                    start=1,
                )
                if item in gold
            ),
            None,
        )
        if (
            first_relevant_rank is not None
            and first_relevant_rank <= top_k
        ):
            reciprocal_rank_sum += (
                1.0 / first_relevant_rank
            )

        ideal_relevant = min(
            len(gold),
            top_k,
        )
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(
                1,
                ideal_relevant + 1,
            )
        )
        dcg = dcg_at_k(
            ranked,
            gold,
            top_k,
        )
        ndcg = (
            dcg / ideal_dcg
            if ideal_dcg > 0
            else 0.0
        )
        ndcg_sum += ndcg

        case_rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "gold_pages": [
                    {
                        "source_id": source_id,
                        "page_number": page_number,
                    }
                    for source_id, page_number
                    in sorted(gold)
                ],
                "first_relevant_rank": (
                    first_relevant_rank
                ),
                "hits": [
                    hit.to_dict()
                    for hit in hits
                ],
            }
        )

    n = len(cases)
    result = {
        "mode": "text_only_bm25",
        "corpus_pages": len(documents),
        "verified_pdf_cases": n,
        "k1": float(config["k1"]),
        "b": float(config["b"]),
        "recall_at_1": recall_hits[1] / n,
        "recall_at_3": recall_hits[3] / n,
        "recall_at_5": recall_hits[5] / n,
        "mrr_at_5": reciprocal_rank_sum / n,
        "ndcg_at_5": ndcg_sum / n,
        "cases": case_rows,
        "scope_note": (
            "Day 3 smoke baseline over one 8-page "
            "datasheet and two verified PDF queries; "
            "not a headline retrieval-quality benchmark."
        ),
    }

    output = ROOT / args.output
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
