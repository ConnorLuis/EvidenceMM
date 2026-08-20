from __future__ import annotations

import argparse
import re
from pathlib import Path

import pymupdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search extracted PDF text and report 1-based pages"
    )
    parser.add_argument("--pdf", required=True)
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="repeat --query for multiple search terms",
    )
    parser.add_argument("--context", type=int, default=120)
    return parser.parse_args()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    args = parse_args()
    pdf = Path(args.pdf).expanduser().resolve()

    with pymupdf.open(pdf) as doc:
        print(f"pdf={pdf}")
        print(f"page_count={len(doc)}")

        for query in args.query:
            print(f"\n=== query: {query} ===")
            hits = 0

            for page_index, page in enumerate(doc):
                text = compact(page.get_text("text"))
                pos = text.lower().find(query.lower())
                if pos < 0:
                    continue

                hits += 1
                lo = max(0, pos - args.context)
                hi = min(len(text), pos + len(query) + args.context)
                snippet = text[lo:hi]

                print(
                    f"page={page_index + 1} "
                    f"snippet={snippet}"
                )

            if hits == 0:
                print("no_hits")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
