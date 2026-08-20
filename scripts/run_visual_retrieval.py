from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from evidencemm.visual_retrieval import (
    encode_query,
    load_model_and_processor,
    rank_scores,
    score_documents,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(
        (ROOT / "configs/visual_retrieval.yaml").read_text(
            encoding="utf-8"
        )
    )
    index = torch.load(
        ROOT / config["index_file"],
        map_location="cpu",
        weights_only=False,
    )

    model, processor, model_load_sec = (
        load_model_and_processor(
            model_name=config["model_name"],
            max_num_visual_tokens=int(
                config["max_num_visual_tokens"]
            ),
        )
    )

    query_embedding, query_sec = encode_query(
        model=model,
        processor=processor,
        query=args.query,
    )
    scores = score_documents(
        processor=processor,
        query_embedding=query_embedding,
        document_embeddings=index["embeddings"],
        device=model.device,
    )

    hits = []
    for rank, (doc_index, score) in enumerate(
        rank_scores(
            scores,
            top_k=int(config["top_k"]),
        ),
        start=1,
    ):
        page = index["pages"][doc_index]
        hits.append(
            {
                "rank": rank,
                "score": score,
                "source_id": page["source_id"],
                "page_number": page["page_number"],
                "image_path": page["image_path"],
            }
        )

    print(
        json.dumps(
            {
                "mode": "vision_only_colqwen2_5",
                "query": args.query,
                "model_load_sec": model_load_sec,
                "query_encode_sec": query_sec,
                "hits": hits,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
