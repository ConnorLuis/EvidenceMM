from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.robot_benchmark_split import (
    ANOMALY_CATEGORY,
    CLEAN_CATEGORY,
    CONFIG_SCHEMA,
    SPLIT_SCHEMA,
    assign_stratified_split,
    build_split_artifact,
    canonical_json_bytes,
    load_eligible_episodes,
    load_gold_event_metadata,
    sha256_path,
    validate_expected_gold_counts,
    validate_expected_source_counts,
    validate_expected_split_counts,
    validate_split_artifact_no_gold_boundaries,
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
        "--split",
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

    split_path = (
        args.split
        if args.split is not None
        else _resolve(
            project_root,
            config["output"]["split_json"],
        )
    )
    if not split_path.is_absolute():
        split_path = project_root / split_path

    loaded = json.loads(
        split_path.read_text(encoding="utf-8")
    )
    if loaded.get("schema_version") != SPLIT_SCHEMA:
        raise ValueError(
            "unexpected Day18 split artifact schema_version"
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

    assignments = assign_stratified_split(
        eligible,
        seed=str(config["protocol"]["seed"]),
        held_out_counts=held_out_counts,
    )
    validate_expected_split_counts(
        assignments,
        config["expected_split"],
    )

    gold_events = load_gold_event_metadata(
        human_gt_path
    )
    validate_expected_gold_counts(
        gold_events,
        config["expected_human_gt"],
    )

    expected = build_split_artifact(
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

    validate_split_artifact_no_gold_boundaries(
        loaded
    )

    if canonical_json_bytes(loaded) != canonical_json_bytes(
        expected
    ):
        raise ValueError(
            "Day18 split artifact differs from deterministic rebuild"
        )

    development_ids = set(
        loaded["splits"]["development"]["episode_ids"]
    )
    held_out_ids = set(
        loaded["splits"]["held_out"]["episode_ids"]
    )
    if development_ids & held_out_ids:
        raise ValueError(
            "development/held-out episode overlap"
        )
    if len(development_ids | held_out_ids) != int(
        config["expected_split"][
            "eligible_episode_count"
        ]
    ):
        raise ValueError(
            "split episode universe differs from expected eligible count"
        )

    summary = {
        "valid": True,
        "benchmark_status": loaded["benchmark_status"],
        "split_scope": loaded["split_scope"],
        "eligible_episode_count": loaded["counts"][
            "eligible_episode_count"
        ],
        "development_episode_count": loaded["counts"][
            "development_episode_count"
        ],
        "held_out_episode_count": loaded["counts"][
            "held_out_episode_count"
        ],
        "development_anomaly_count": loaded["counts"][
            "development_anomaly_count"
        ],
        "held_out_anomaly_count": loaded["counts"][
            "held_out_anomaly_count"
        ],
        "gold_metadata_counts_by_split": loaded[
            "gold_metadata_counts_by_split"
        ],
        "human_gt_used_for_membership": loaded[
            "anti_leakage"
        ]["human_gt_used_for_membership"],
        "failure_interval_used_for_membership": loaded[
            "anti_leakage"
        ]["failure_interval_used_for_membership"],
        "split_artifact_contains_failure_boundaries": loaded[
            "anti_leakage"
        ]["split_artifact_contains_failure_boundaries"],
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
