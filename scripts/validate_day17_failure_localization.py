from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.data_binding import sha256_file
from evidencemm.failure_localization_eval import (
    CandidateFrame,
    build_day17_report,
    load_human_gt_events,
    validate_gold_universe,
)
from evidencemm.state_action_selection import (
    load_state_action_samples,
)


CONFIG_SCHEMA = "evidencemm_day17_failure_localization_eval_v1"


def _resolve(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _load_candidate_sets(
    report: dict,
) -> dict[str, list[CandidateFrame]]:
    result = {}

    for episode_id, rows in report[
        "candidate_sets"
    ].items():
        candidates = []
        seen = set()

        for row in rows:
            frame_index = int(row["frame_index"])
            if frame_index in seen:
                raise ValueError(
                    f"{episode_id}: duplicate candidate frame"
                )
            seen.add(frame_index)

            candidates.append(
                CandidateFrame(
                    frame_index=frame_index,
                    timestamp_sec=float(
                        row["timestamp_sec"]
                    ),
                    reasons=tuple(row["reasons"]),
                    metrics={
                        str(key): float(value)
                        for key, value in row[
                            "metrics"
                        ].items()
                    },
                )
            )

        result[str(episode_id)] = candidates

    return result


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
        "--report",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dataset-root",
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

    report_path = (
        args.report
        if args.report is not None
        else _resolve(
            project_root,
            config["output"]["report_json"],
        )
    )
    if not report_path.is_absolute():
        report_path = project_root / report_path

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    if report.get(
        "gold_read_during_candidate_generation"
    ) is not False:
        raise ValueError(
            "Day17 anti-leakage flag must be false"
        )

    selector_config_path = _resolve(
        project_root,
        config["selector"]["config_path"],
    )
    if report.get("selector_config_sha256") != (
        sha256_file(selector_config_path)
    ):
        raise ValueError(
            "selector config SHA256 differs from report"
        )

    inputs = config["inputs"]
    expected = config["expected"]
    expected_episode_ids = [
        str(value)
        for value in expected["expected_episode_ids"]
    ]

    candidate_sets = _load_candidate_sets(report)
    if set(candidate_sets) != set(
        expected_episode_ids
    ):
        raise ValueError(
            "candidate-set episode universe mismatch"
        )

    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else _resolve(
            project_root,
            inputs["dataset_root"],
        ).resolve()
    )

    max_selected_frames = int(
        yaml.safe_load(
            selector_config_path.read_text(
                encoding="utf-8"
            )
        )["selection"]["max_selected_frames"]
    )

    for episode_id in expected_episode_ids:
        candidates = candidate_sets[episode_id]
        if not candidates:
            raise ValueError(
                f"{episode_id}: empty candidate set"
            )
        if len(candidates) > max_selected_frames:
            raise ValueError(
                f"{episode_id}: candidate count exceeds "
                "frozen selector budget"
            )

        samples = load_state_action_samples(
            dataset_root
            / episode_id
            / "samples.csv",
            verify_tracking_error=True,
        )
        by_frame = {
            int(sample.frame_index): sample
            for sample in samples
        }

        for candidate in candidates:
            if candidate.frame_index not in by_frame:
                raise ValueError(
                    f"{episode_id}: candidate frame "
                    "not present in samples"
                )
            expected_ts = float(
                by_frame[
                    candidate.frame_index
                ].timestamp_sec
            )
            if abs(
                candidate.timestamp_sec
                - expected_ts
            ) > 1e-9:
                raise ValueError(
                    f"{episode_id}: candidate timestamp "
                    "does not bind to canonical sample"
                )

    gold_events = load_human_gt_events(
        _resolve(
            project_root,
            inputs["human_gt_jsonl"],
        )
    )

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

    expected_report = build_day17_report(
        candidate_frames_by_episode=candidate_sets,
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

    if report != expected_report:
        raise ValueError(
            "Day17 report metrics do not reproduce "
            "from committed GT and reported candidate sets"
        )

    human_gt_summary = json.loads(
        _resolve(
            project_root,
            inputs["human_gt_summary"],
        ).read_text(encoding="utf-8")
    )
    if (
        int(human_gt_summary["verified_count"])
        != int(expected["verified_event_count"])
        or int(
            human_gt_summary[
                "reviewed_unresolved_count"
            ]
        )
        != int(
            expected["reviewed_unresolved_count"]
        )
        or human_gt_summary[
            "human_review_complete"
        ]
        is not True
    ):
        raise ValueError(
            "Day16 human-GT summary differs from "
            "Day17 expected contract"
        )

    print(
        json.dumps(
            {
                "valid": True,
                "benchmark_status": report[
                    "benchmark_status"
                ],
                "verified_event_count": report[
                    "metrics"
                ]["verified_event_count"],
                "reviewed_unresolved_count": report[
                    "metrics"
                ]["reviewed_unresolved_count"],
                "overall": report["metrics"]["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
