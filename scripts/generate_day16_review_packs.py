from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.review_pack import (
    SelectionConfig,
    generate_review_packs,
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

    selection = SelectionConfig(
        **config["selection"]
    )
    selection.validate()

    manifest = generate_review_packs(
        project_root=ROOT,
        audit_path=_resolve(
            config["inputs"][
                "source_audit_jsonl"
            ]
        ),
        review_cases_path=_resolve(
            config["inputs"][
                "anomaly_review_jsonl"
            ]
        ),
        output_root=_resolve(
            config["output"]["root"]
        ),
        config=selection,
        binding_report_path=_resolve(
            config["inputs"][
                "diagnostic_binding_report"
            ]
        ),
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
