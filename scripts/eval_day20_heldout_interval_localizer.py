from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from evidencemm.failure_localization_eval import (
    build_frozen_review_candidates,
)
from evidencemm.heldout_interval_eval import (
    CONFIG_SCHEMA,
    build_report,
    load_heldout_gold_events,
    sha256_path,
    validate_expected_heldout_gold,
    validate_frozen_day19_model,
)
from evidencemm.interval_localizer import (
    build_interval_proposals,
    load_day18_split,
    split_episode_sets,
)
from evidencemm.review_pack import SelectionConfig
from evidencemm.temporal_evidence import EpisodeManifest


def _resolve(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _git_blob_sha1(
    project_root: Path,
    repository_path: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            f"HEAD:{repository_path}",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _assert_worktree_unchanged(
    project_root: Path,
    repository_path: str,
) -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            repository_path,
        ],
        cwd=project_root,
    )
    if result.returncode != 0:
        raise ValueError(
            f"frozen input modified in worktree: {repository_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/day20_heldout_interval_eval.yaml"
        ),
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
            "unexpected Day20 config schema_version"
        )

    inputs = config["inputs"]
    expected = config["expected"]

    split_path = _resolve(
        project_root,
        inputs["benchmark_split_json"],
    )
    model_path = _resolve(
        project_root,
        inputs["interval_model_json"],
    )
    human_gt_path = _resolve(
        project_root,
        inputs["human_gt_jsonl"],
    )
    selector_config_path = _resolve(
        project_root,
        inputs["frozen_selector_config"],
    )

    model_repository_path = str(
        inputs["interval_model_json"]
    )
    _assert_worktree_unchanged(
        project_root,
        model_repository_path,
    )
    model_blob_sha1 = _git_blob_sha1(
        project_root,
        model_repository_path,
    )
    if model_blob_sha1 != str(
        config["provenance"][
            "expected_day19_model_blob_sha1"
        ]
    ):
        raise ValueError(
            "Day19 model Git blob differs from "
            "the frozen Day20 expectation"
        )

    split_artifact = load_day18_split(
        split_path
    )
    (
        development_ids,
        held_out_ids,
        development_anomaly_ids,
        held_out_anomaly_ids,
    ) = split_episode_sets(
        split_artifact
    )

    if len(held_out_ids) != int(
        expected["held_out_episode_count"]
    ):
        raise ValueError(
            "held-out episode count mismatch"
        )
    if len(held_out_anomaly_ids) != int(
        expected[
            "held_out_anomaly_episode_count"
        ]
    ):
        raise ValueError(
            "held-out anomaly episode count mismatch"
        )

    wanted_held_out_anomaly_ids = {
        str(value)
        for value in expected[
            "held_out_anomaly_episode_ids"
        ]
    }
    if held_out_anomaly_ids != (
        wanted_held_out_anomaly_ids
    ):
        raise ValueError(
            "held-out anomaly episode IDs mismatch"
        )

    model = json.loads(
        model_path.read_text(
            encoding="utf-8"
        )
    )
    selected_radius = validate_frozen_day19_model(
        model,
        expected_selected_radius_frames=int(
            expected["selected_radius_frames"]
        ),
        split_sha256=sha256_path(
            split_path
        ),
        selector_config_sha256=sha256_path(
            selector_config_path
        ),
        human_gt_sha256=sha256_path(
            human_gt_path
        ),
    )

    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else _resolve(
            project_root,
            inputs["dataset_root"],
        ).resolve()
    )

    selector_raw = yaml.safe_load(
        selector_config_path.read_text(
            encoding="utf-8"
        )
    )
    selection_config = SelectionConfig(
        **selector_raw["selection"]
    )
    selection_config.validate()

    manifest_root = _resolve(
        project_root,
        inputs["diagnostic_manifest_root"],
    )
    processed_root = _resolve(
        project_root,
        inputs["processed_root"],
    )

    # Held-out prediction phase:
    # generate candidates/proposals first, with the frozen Day19 radius.
    # No Human-GT row has been loaded yet.
    proposals_by_episode = {}

    for episode_id in sorted(
        held_out_anomaly_ids
    ):
        if episode_id in development_ids:
            raise ValueError(
                "development/held-out split overlap"
            )

        manifest_path = (
            manifest_root
            / f"{episode_id}.json"
        )
        manifest = EpisodeManifest.model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        candidates = build_frozen_review_candidates(
            project_root=project_root,
            dataset_root=dataset_root,
            episode_id=episode_id,
            diagnostic_manifest_path=manifest_path,
            frame_records_path=(
                processed_root
                / episode_id
                / "frames.jsonl"
            ),
            selection_config=selection_config,
        )

        proposals_by_episode[episode_id] = (
            build_interval_proposals(
                candidates,
                frame_count=int(
                    manifest.frame_count
                ),
                radius_frames=selected_radius,
            )
        )

    if set(proposals_by_episode) != (
        held_out_anomaly_ids
    ):
        raise ValueError(
            "held-out proposal universe mismatch"
        )

    # Only now materialize held-out interval GT.
    gold_events = load_heldout_gold_events(
        human_gt_path,
        allowed_episode_ids=(
            held_out_anomaly_ids
        ),
    )
    validate_expected_heldout_gold(
        gold_events,
        expected_event_count=int(
            expected["held_out_event_count"]
        ),
        expected_verified_count=int(
            expected[
                "held_out_verified_event_count"
            ]
        ),
        expected_unresolved_count=int(
            expected[
                "held_out_reviewed_unresolved_count"
            ]
        ),
        expected_event_ids=expected[
            "held_out_event_ids"
        ],
    )

    report = build_report(
        frozen_after_day19_commit=str(
            config["provenance"][
                "frozen_after_day19_commit"
            ]
        ),
        day19_model_blob_sha1=(
            model_blob_sha1
        ),
        split_sha256=sha256_path(
            split_path
        ),
        model_sha256=sha256_path(
            model_path
        ),
        selector_config_sha256=sha256_path(
            selector_config_path
        ),
        human_gt_sha256=sha256_path(
            human_gt_path
        ),
        selected_radius_frames=(
            selected_radius
        ),
        proposals_by_episode=(
            proposals_by_episode
        ),
        gold_events=gold_events,
        iou_thresholds=[
            float(value)
            for value in config[
                "evaluation"
            ]["iou_thresholds"]
        ],
    )

    output_path = _resolve(
        project_root,
        config["output"]["report_json"],
    )
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
    )

    print(
        json.dumps(
            {
                "schema_version": report[
                    "schema_version"
                ],
                "evaluation_status": report[
                    "evaluation_status"
                ],
                "evaluation_split": report[
                    "evaluation_split"
                ],
                "selected_radius_frames": (
                    selected_radius
                ),
                "held_out_anomaly_episode_count": (
                    len(held_out_anomaly_ids)
                ),
                "held_out_verified_event_count": (
                    report["held_out_universe"][
                        "verified_event_count"
                    ]
                ),
                "model_selection_performed": (
                    report["anti_leakage"][
                        "model_selection_performed"
                    ]
                ),
                "radius_tuned_on_held_out": (
                    report["anti_leakage"][
                        "radius_tuned_on_held_out"
                    ]
                ),
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
