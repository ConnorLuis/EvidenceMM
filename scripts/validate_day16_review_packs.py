from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.review_pack import (
    validate_review_pack_output,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/day16_review_pack.yaml",
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    report = validate_review_pack_output(
        _resolve(
            config["output"]["root"]
        ),
        expected_episode_ids=list(
            config["expected_episode_ids"]
        ),
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
