from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.data_binding import sha256_file
from evidencemm.failure_localization_eval import (
    build_day17_report,
    build_frozen_review_candidates,
    load_human_gt_events,
    validate_gold_universe,
)
from evidencemm.review_pack import SelectionConfig


CONFIG_SCHEMA = "evidencemm_day17_failure_localization_eval_v1"


def _resolve(
    project_root: Path,
    value: str | Path,
) -> Path:
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
            "configs/day17_failure_localization_eval.yaml"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
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
            "unexpected Day17 config schema_version"
        )

    inputs = config["inputs"]
    expected = config["expected"]

    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else _resolve(
            project_root,
            inputs["dataset_root"],
        ).resolve()
    )

    selector_config_path = _resolve(
        project_root,
        config["selector"]["config_path"],
    )
    selector_config_raw = yaml.safe_load(
        selector_config_path.read_text(
            encoding="utf-8"
        )
    )
    selection_config = SelectionConfig(
        **selector_config_raw["selection"]
    )
    selection_config.validate()

    diagnostic_manifest_root = _resolve(
        project_root,
        inputs["diagnostic_manifest_root"],
    )
    processed_root = _resolve(
        project_root,
        inputs["processed_root"],
    )

    expected_episode_ids = [
        str(value)
        for value in expected["expected_episode_ids"]
    ]
    if len(expected_episode_ids) != len(
        set(expected_episode_ids)
    ):
        raise ValueError(
            "duplicate episode id in Day17 config"
        )

    # Anti-leakage boundary:
    # build every candidate set from frozen source/config inputs
    # before loading any human-reviewed interval labels.
    candidate_frames_by_episode = {}
    for episode_id in expected_episode_ids:
        candidate_frames_by_episode[episode_id] = (
            build_frozen_review_candidates(
                project_root=project_root,
                dataset_root=dataset_root,
                episode_id=episode_id,
                diagnostic_manifest_path=(
                    diagnostic_manifest_root
                    / f"{episode_id}.json"
                ),
                frame_records_path=(
                    processed_root
                    / episode_id
                    / "frames.jsonl"
                ),
                selection_config=selection_config,
            )
        )

    gt_path = _resolve(
        project_root,
        inputs["human_gt_jsonl"],
    )
    gold_events = load_human_gt_events(gt_path)

    validate_gold_universe(
        gold_events,
        expected_episode_count=int(
            expected["episode_count"]
        ),
        expected_event_count=int(
            expected["event_count"]
        ),
        expected_verified_count=int(
            expected["verified_event_count"]
        ),
        expected_unresolved_count=int(
            expected["reviewed_unresolved_count"]
        ),
    )

    report = build_day17_report(
        candidate_frames_by_episode=(
            candidate_frames_by_episode
        ),
        gold_events=gold_events,
        tolerance_frames=(
            config["evaluation"]["tolerance_frames"]
        ),
        selector_config_path=(
            Path(config["selector"]["config_path"])
            .as_posix()
        ),
        selector_config_sha256=sha256_file(
            selector_config_path
        ),
    )

    output_path = (
        args.output
        if args.output is not None
        else _resolve(
            project_root,
            config["output"]["report_json"],
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
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
