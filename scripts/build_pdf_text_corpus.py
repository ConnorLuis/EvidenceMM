from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencemm.pdf_corpus import (
    load_source_manifest,
    standardize_pdf_pages,
)
from evidencemm.text_retrieval import save_corpus


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize a bound PDF into page-level "
            "text documents."
        )
    )
    parser.add_argument(
        "--source-id",
        default="sts3215_datasheet",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/processed/text/"
            "sts3215_pages.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = (
        ROOT
        / "data"
        / "manifests"
        / "sources"
        / f"{args.source_id}.json"
    )

    manifest = load_source_manifest(
        manifest_path
    )
    documents = standardize_pdf_pages(
        manifest,
        project_root=ROOT,
    )

    output = save_corpus(
        documents,
        ROOT / args.output,
    )

    payload = {
        "source_id": manifest.source_id,
        "page_count": len(documents),
        "nonempty_pages": sum(
            bool(doc.text)
            for doc in documents
        ),
        "total_chars": sum(
            doc.char_count
            for doc in documents
        ),
        "output": str(
            output.relative_to(ROOT)
        ),
    }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
