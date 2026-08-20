from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencemm.data_binding import bind_source
from evidencemm.schemas import SourceType


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind one real PDF/image asset to an EvidenceMM source id"
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--type",
        required=True,
        choices=["pdf", "image"],
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--origin-uri")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_type = SourceType(args.type)

    manifest = bind_source(
        source_id=args.source_id,
        source_type=source_type,
        path=args.path,
        origin_uri=args.origin_uri,
        project_root=ROOT,
    )

    output = (
        Path(args.output)
        if args.output
        else ROOT
        / "data"
        / "manifests"
        / "sources"
        / f"{args.source_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    ))
    print(f"manifest={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
