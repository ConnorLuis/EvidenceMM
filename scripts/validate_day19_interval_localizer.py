from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from evidencemm.failure_localization_eval import (
    build_frozen_review_candidates,
)
from evidencemm.interval_localizer import (
    CONFIG_SCHEMA,
    MODEL_SCHEMA,
    REPORT_SCHEMA,
    build_development_report,
    build_model_artifact,
    canonical_json_bytes,
    load_day18_split,
    load_development_gold_events,
    select_radius,
    split_episode_sets,
    validate_expected_development_gold,
    validate_expected_split_counts,
    validate_model_artifact_no_gold_boundaries,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/day19_interval_localizer.yaml"
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
        config_path.read_text(
            encoding="utf-8"
        )
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(
            "unexpected Day19 config schema_version"
        )

    inputs = config["inputs"]
    expected = config["expected"]

    split_path = _resolve(
        project_root,
        inputs["benchmark_split_json"],
    )
    split_artifact = load_day18_split(
        split_path
    )
    validate_expected_split_counts(
        split_artifact,
        expected,
    )
    (
        development_ids,
        held_out_ids,
        development_anomaly_ids,
        held_out_anomaly_ids,
    ) = split_episode_sets(
        split_artifact
    )

    model_path = _resolve(
        project_root,
        config["output"]["model_json"],
    )
    report_path = _resolve(
        project_root,
        config["output"][
            "development_report_json"
        ],
    )
    loaded_model = json.loads(
        model_path.read_text(
            encoding="utf-8"
        )
    )
    loaded_report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if loaded_model.get(
        "schema_version"
    ) != MODEL_SCHEMA:
        raise ValueError(
            "unexpected Day19 model schema_version"
        )
    if loaded_report.get(
        "schema_version"
    ) != REPORT_SCHEMA:
        raise ValueError(
            "unexpected Day19 report schema_version"
        )

    validate_model_artifact_no_gold_boundaries(
        loaded_model
    )

    model_text = json.dumps(
        loaded_model,
        ensure_ascii=False,
        sort_keys=True,
    )
    report_text = json.dumps(
        loaded_report,
        ensure_ascii=False,
        sort_keys=True,
    )
    for episode_id in held_out_ids:
        if episode_id in model_text:
            raise ValueError(
                "held-out episode ID leaked into Day19 model"
            )
        if episode_id in report_text:
            raise ValueError(
                "held-out episode ID leaked into "
                "Day19 development report"
            )

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
        inputs["frozen_selector_config"],
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

    candidate_sets = {}
    frame_counts = {}
    for episode_id in sorted(
        development_anomaly_ids
    ):
        manifest_path = (
            manifest_root / f"{episode_id}.json"
        )
        manifest = EpisodeManifest.model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
        frame_counts[episode_id] = int(
            manifest.frame_count
        )
        candidate_sets[episode_id] = (
            build_frozen_review_candidates(
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
        )

    human_gt_path = _resolve(
        project_root,
        inputs["human_gt_jsonl"],
    )
    development_events = (
        load_development_gold_events(
            human_gt_path,
            allowed_episode_ids=(
                development_anomaly_ids
            ),
        )
    )
    validate_expected_development_gold(
        development_events,
        expected,
    )
    verified_events = [
        event
        for event in development_events
        if event.is_verified
    ]
    unresolved_events = [
        event
        for event in development_events
        if not event.is_verified
    ]

    radius_grid = [
        int(value)
        for value in config["localizer"][
            "radius_grid_frames"
        ]
    ]
    iou_thresholds = [
        float(value)
        for value in config["localizer"][
            "iou_thresholds"
        ]
    ]

    (
        selected_radius,
        grid_metrics,
        selected_results,
    ) = select_radius(
        radius_grid_frames=radius_grid,
        candidate_sets=candidate_sets,
        frame_counts=frame_counts,
        verified_events=verified_events,
        iou_thresholds=iou_thresholds,
    )

    expected_model = build_model_artifact(
        split_sha256=_sha256(split_path),
        selector_config_sha256=_sha256(
            selector_config_path
        ),
        human_gt_sha256=_sha256(
            human_gt_path
        ),
        frozen_after_day18_commit=str(
            config["provenance"][
                "frozen_after_day18_commit"
            ]
        ),
        development_episode_count=len(
            development_ids
        ),
        development_anomaly_episode_count=len(
            development_anomaly_ids
        ),
        held_out_episode_count=len(
            held_out_ids
        ),
        held_out_anomaly_episode_count=len(
            held_out_anomaly_ids
        ),
        development_verified_event_count=len(
            verified_events
        ),
        development_reviewed_unresolved_count=len(
            unresolved_events
        ),
        radius_grid_frames=radius_grid,
        iou_thresholds=iou_thresholds,
        selected_radius_frames=selected_radius,
        grid_metrics=grid_metrics,
    )

    expected_report = build_development_report(
        selected_radius_frames=selected_radius,
        grid_metrics=grid_metrics,
        selected_results=selected_results,
        unresolved_events=unresolved_events,
        candidate_sets=candidate_sets,
        frame_counts=frame_counts,
        iou_thresholds=iou_thresholds,
    )

    if canonical_json_bytes(
        loaded_model
    ) != canonical_json_bytes(
        expected_model
    ):
        raise ValueError(
            "Day19 model artifact differs from "
            "deterministic development-only rebuild"
        )
    if canonical_json_bytes(
        loaded_report
    ) != canonical_json_bytes(
        expected_report
    ):
        raise ValueError(
            "Day19 development report differs from "
            "deterministic rebuild"
        )

    selected_metrics = loaded_model[
        "development_selection"
    ]["selected_metrics"]

    print(
        json.dumps(
            {
                "valid": True,
                "model_status": loaded_model[
                    "model_status"
                ],
                "model_selection_split": loaded_model[
                    "model_selection_split"
                ],
                "selected_radius_frames": loaded_model[
                    "localizer"
                ]["selected_radius_frames"],
                "development_verified_event_count": (
                    len(verified_events)
                ),
                "development_reviewed_unresolved_count": (
                    len(unresolved_events)
                ),
                "held_out_episode_count_not_evaluated": (
                    len(held_out_ids)
                ),
                "held_out_anomaly_episode_count_not_evaluated": (
                    len(held_out_anomaly_ids)
                ),
                "held_out_gt_used_for_model_selection": (
                    loaded_model["anti_leakage"][
                        "held_out_gt_used_for_model_selection"
                    ]
                ),
                "held_out_metrics_reported": (
                    loaded_model["anti_leakage"][
                        "held_out_metrics_reported"
                    ]
                ),
                "selected_metrics": selected_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
