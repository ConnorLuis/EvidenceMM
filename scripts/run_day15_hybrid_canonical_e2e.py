import json


def main():
    payload = {
        "pipeline": "hybrid_retrieval_canonical_e2e",
        "document_retriever": "bm25+dense",
        "fusion": "rrf",
        "generation_connected": False,
        "status": "day15_hybrid_layer"
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
