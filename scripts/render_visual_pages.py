from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.pdf_corpus import load_source_manifest
from evidencemm.visual_corpus import (
    render_pdf_pages,
    save_visual_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-id",
        default="sts3215_datasheet",
    )
    parser.add_argument(
        "--config",
        default="configs/visual_retrieval.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(
        (ROOT / args.config).read_text(encoding="utf-8")
    )

    manifest = load_source_manifest(
        ROOT
        / "data"
        / "manifests"
        / "sources"
        / f"{args.source_id}.json"
    )

    pages = render_pdf_pages(
        manifest,
        project_root=ROOT,
        output_dir=config["page_dir"],
        render_dpi=int(config["render_dpi"]),
    )

    output = ROOT / config["page_manifest"]
    save_visual_manifest(pages, output)

    print(
        json.dumps(
            {
                "source_id": manifest.source_id,
                "page_count": len(pages),
                "render_dpi": int(config["render_dpi"]),
                "page_manifest": str(output.relative_to(ROOT)),
                "pages": [
                    {
                        "page_number": p.page_number,
                        "width_px": p.width_px,
                        "height_px": p.height_px,
                        "sha12": p.image_sha256[:12],
                    }
                    for p in pages
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
