from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from evidencemm.hybrid_retrieval import RankedPage, fuse_rrf
from evidencemm.text_retrieval import BM25Index, load_corpus
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

    text_documents = load_corpus(ROOT / config["text_corpus"])
    text_index = BM25Index(
        text_documents,
        k1=float(text_config["k1"]),
        b=float(text_config["b"]),
    )
    text_raw_hits = text_index.search(
        args.query,
        top_k=int(config["top_k"]),
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
    query_embedding, query_encode_sec = encode_query(
        model=model,
        processor=processor,
        query=args.query,
    )
    visual_scores = score_documents(
        processor=processor,
        query_embedding=query_embedding,
        document_embeddings=visual_index["embeddings"],
        device=model.device,
    )
    visual_ranking = rank_scores(
        visual_scores,
        top_k=int(config["top_k"]),
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

    fused = fuse_rrf(
        text_hits=text_hits,
        vision_hits=vision_hits,
        top_k=int(config["top_k"]),
        rrf_k=int(config["rrf_k"]),
        text_weight=float(config["text_weight"]),
        vision_weight=float(config["vision_weight"]),
    )

    print(
        json.dumps(
            {
                "mode": "hybrid_rrf",
                "query": args.query,
                "rrf_k": int(config["rrf_k"]),
                "text_weight": float(config["text_weight"]),
                "vision_weight": float(config["vision_weight"]),
                "model_load_sec": model_load_sec,
                "query_encode_sec": query_encode_sec,
                "text_hits": [hit.__dict__ for hit in text_hits],
                "vision_hits": [hit.__dict__ for hit in vision_hits],
                "hybrid_hits": [hit.to_dict() for hit in fused],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
