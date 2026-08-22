from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.robot_benchmark_split import (
    ANOMALY_CATEGORY,
    CLEAN_CATEGORY,
    CONFIG_SCHEMA,
    assign_stratified_split,
    build_split_artifact,
    load_eligible_episodes,
    load_gold_event_metadata,
    sha256_path,
    validate_expected_gold_counts,
    validate_expected_source_counts,
    validate_expected_split_counts,
)


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/day18_robot_benchmark_split.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    config_path = _resolve(
        project_root,
        args.config,
    )
    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(
            "unexpected Day18 split config schema_version"
        )

    source_audit_path = _resolve(
        project_root,
        config["inputs"]["source_audit_jsonl"],
    )
    human_gt_path = _resolve(
        project_root,
        config["inputs"]["human_gt_jsonl"],
    )

    rows, eligible = load_eligible_episodes(
        source_audit_path
    )
    validate_expected_source_counts(
        rows,
        config["expected_source_audit"],
    )

    held_out_counts = {
        CLEAN_CATEGORY: int(
            config["protocol"]["held_out_counts"][
                CLEAN_CATEGORY
            ]
        ),
        ANOMALY_CATEGORY: int(
            config["protocol"]["held_out_counts"][
                ANOMALY_CATEGORY
            ]
        ),
    }

    # Membership is frozen from source-audit episode identity/category only.
    # Human GT is deliberately not loaded until after assignments exist.
    assignments = assign_stratified_split(
        eligible,
        seed=str(config["protocol"]["seed"]),
        held_out_counts=held_out_counts,
    )
    validate_expected_split_counts(
        assignments,
        config["expected_split"],
    )

    # Post-membership metadata validation only. No interval values are
    # materialized into the split artifact.
    gold_events = load_gold_event_metadata(
        human_gt_path
    )
    validate_expected_gold_counts(
        gold_events,
        config["expected_human_gt"],
    )

    artifact = build_split_artifact(
        eligible_episodes=eligible,
        assignments=assignments,
        gold_events=gold_events,
        source_audit_path=Path(
            config["inputs"]["source_audit_jsonl"]
        ).as_posix(),
        source_audit_sha256=sha256_path(
            source_audit_path
        ),
        human_gt_path=Path(
            config["inputs"]["human_gt_jsonl"]
        ).as_posix(),
        human_gt_sha256=sha256_path(
            human_gt_path
        ),
        seed=str(config["protocol"]["seed"]),
        held_out_counts=held_out_counts,
        frozen_after_day17_commit=str(
            config["provenance"][
                "frozen_after_day17_commit"
            ]
        ),
    )

    output_path = (
        args.output
        if args.output is not None
        else _resolve(
            project_root,
            config["output"]["split_json"],
        )
    )
    if not output_path.is_absolute():
        output_path = project_root / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
