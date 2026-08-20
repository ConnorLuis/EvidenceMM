from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.text_retrieval import (
    BM25Index,
    load_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Day 3 page-level "
            "text-only BM25 baseline."
        )
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--corpus",
        default=(
            "data/processed/text/"
            "sts3215_pages.jsonl"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/text_retrieval.yaml",
    )
    parser.add_argument(
        "--top-k",
        type=int,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = yaml.safe_load(
        (ROOT / args.config).read_text(
            encoding="utf-8"
        )
    )
    top_k = (
        args.top_k
        if args.top_k is not None
        else int(config["top_k"])
    )

    documents = load_corpus(
        ROOT / args.corpus
    )
    index = BM25Index(
        documents,
        k1=float(config["k1"]),
        b=float(config["b"]),
    )
    hits = index.search(
        args.query,
        top_k=top_k,
    )

    print(
        json.dumps(
            {
                "mode": "text_only_bm25",
                "query": args.query,
                "top_k": top_k,
                "hits": [
                    hit.to_dict()
                    for hit in hits
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
