#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencemm.day29_blind_review_pack import (
    build_pack,
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "day29_blind_review_pack.yaml"
        ),
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = (
        ROOT
        / args.config
    ).resolve()

    if args.preflight:
        result = preflight(
            project_root=ROOT,
            config_path=config_path,
        )
    else:
        result = build_pack(
            project_root=ROOT,
            config_path=config_path,
        )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
