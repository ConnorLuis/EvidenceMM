from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.temporal_evidence import (
    load_frame_records,
)
from evidencemm.temporal_eval import (
    evaluate_event_coverage,
    load_annotations,
)
from evidencemm.temporal_slicing import (
    load_temporal_slices,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-id",
        required=True,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )

    processed_dir = (
        ROOT
        / config["processed_root"]
        / args.episode_id
    )

    records = load_frame_records(
        processed_dir / "frames.jsonl"
    )
    slices = load_temporal_slices(
        processed_dir
        / "temporal_slices.jsonl"
    )

    annotations = load_annotations(
        ROOT
        / "data/eval"
        / (
            "day7_temporal_events_"
            f"{args.episode_id}.jsonl"
        )
    )

    result = evaluate_event_coverage(
        annotations=annotations,
        records=records,
        slices=slices,
    )

    payload = {
        "mode":
            "uniform_midpoint_temporal_event_coverage",
        "episode_id":
            args.episode_id,
        "slice_duration_sec":
            float(config["slice_duration_sec"]),
        "timestamp_source":
            config["timestamp_source"],
        **result,
        "scope_note": (
            "Day 7 temporal smoke baseline over one "
            "human-annotated SO-ARM101 episode. "
            "Coverage measures whether a uniform "
            "two-second slice midpoint falls inside "
            "each verified event interval."
        ),
    }

    output = (
        ROOT
        / "reports"
        / (
            "day7_temporal_event_coverage_"
            f"{args.episode_id}.json"
        )
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
