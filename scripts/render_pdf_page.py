from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one 1-based PDF page to PNG for human verification"
    )
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--page", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf = Path(args.pdf).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if args.page < 1:
        raise SystemExit("--page is 1-based and must be >= 1")

    output.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf) as doc:
        if args.page > len(doc):
            raise SystemExit(
                f"page {args.page} exceeds page_count={len(doc)}"
            )

        page = doc.load_page(args.page - 1)
        scale = args.dpi / 72.0
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            alpha=False,
        )
        pix.save(output)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
