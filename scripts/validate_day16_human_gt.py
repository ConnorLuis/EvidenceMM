from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencemm.day16_human_gt import (
    validate_promoted_human_gt,
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path(
            "reports/day16_review_pack"
        ),
    )

    parser.add_argument(
        "--source-cases",
        type=Path,
        default=Path(
            "data/annotations/"
            "day16_anomaly_review_cases.jsonl"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/mnt/f/episodes_pick_place_pilot_v5"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/annotations/"
            "day16_human_gt_events.jsonl"
        ),
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/annotations/"
            "day16_human_gt_summary.json"
        ),
    )

    args = parser.parse_args()

    summary = validate_promoted_human_gt(
        review_root=args.review_root,
        source_cases_path=args.source_cases,
        dataset_root=args.dataset_root,
        output_path=args.output,
        summary_path=args.summary,
    )

    print(
        json.dumps(
            {
                "valid": True,
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
