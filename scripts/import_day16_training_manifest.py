from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.robot_failure_dataset import (
    AuditCategory,
    audit_training_row,
    build_anomaly_review_case,
    load_training_manifest,
    save_jsonl,
    source_presence_complete,
    required_anomaly_source_presence_complete,
    summarize_source_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "day16_robot_failure_data.yaml"
        ),
    )
    parser.add_argument(
        "--training-manifest",
        default=None,
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
    )
    args = parser.parse_args()

    config_path = Path(
        args.config
    )
    if not config_path.is_absolute():
        config_path = (
            ROOT / config_path
        )
    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    source = config["source"]

    manifest_path = Path(
        args.training_manifest
        or source["training_manifest"]
    ).expanduser()
    dataset_root = Path(
        args.dataset_root
        or source["dataset_root"]
    ).expanduser()

    rows = load_training_manifest(
        manifest_path,
        encoding=str(
            source["encoding"]
        ),
    )

    audit = [
        audit_training_row(
            row,
            dataset_root=dataset_root,
        )
        for row in rows
    ]

    binding = config[
        "diagnostic_binding"
    ]
    anomaly_cases = [
        build_anomaly_review_case(
            item,
            diagnostic_manifest_root=(
                binding["manifest_root"]
            ),
            diagnostic_processed_root=(
                binding["processed_root"]
            ),
        )
        for item in audit
        if (
            item.audit_category
            == AuditCategory.OPERATION_ANOMALY
        )
    ]

    audit_config = config["audit"]

    audit_path = (
        ROOT
        / audit_config[
            "output_jsonl"
        ]
    )
    review_path = (
        ROOT
        / audit_config[
            "anomaly_review_jsonl"
        ]
    )
    report_path = (
        ROOT
        / audit_config[
            "report_json"
        ]
    )

    save_jsonl(
        audit,
        audit_path,
    )
    save_jsonl(
        anomaly_cases,
        review_path,
    )

    summary = summarize_source_audit(
        audit
    )

    expected = config[
        "expected_current_source_audit"
    ]
    actual_categories = summary[
        "categories"
    ]
    expected_checks = {
        "total_rows": (
            summary["total_rows"]
            == int(expected["total_rows"])
        ),
        "clean_reference_candidate": (
            actual_categories.get(
                "clean_reference_candidate",
                0,
            )
            == int(
                expected[
                    "clean_reference_candidate"
                ]
            )
        ),
        "operation_anomaly": (
            actual_categories.get(
                "operation_anomaly",
                0,
            )
            == int(
                expected[
                    "operation_anomaly"
                ]
            )
        ),
        "demo_quality_only": (
            actual_categories.get(
                "demo_quality_only",
                0,
            )
            == int(
                expected[
                    "demo_quality_only"
                ]
            )
        ),
        "technical_exclusion": (
            actual_categories.get(
                "technical_exclusion",
                0,
            )
            == int(
                expected[
                    "technical_exclusion"
                ]
            )
        ),
        "manual_review_required": (
            actual_categories.get(
                "manual_review_required",
                0,
            )
            == int(
                expected[
                    "manual_review_required"
                ]
            )
        ),
    }

    expected_ids = list(
        config[
            "expected_operation_anomaly_episode_ids"
        ]
    )
    actual_ids = [
        item.episode_id
        for item in anomaly_cases
    ]
    anomaly_ids_match = (
        actual_ids == expected_ids
    )

    all_source_presence_complete = all(
        source_presence_complete(item)
        for item in audit
    )

    anomaly_source_presence_complete = (
        required_anomaly_source_presence_complete(
            audit
        )
    )

    technical_exclusion_missing_sources = [
        {
            "episode_id": item.episode_id,
            "episode_dir": item.raw_episode_dir_exists,
            "metadata": item.metadata_exists,
            "samples_csv": item.samples_csv_exists,
            "front": item.front_dir_exists,
            "wrist": item.wrist_dir_exists,
        }
        for item in audit
        if (
            item.audit_category
            == AuditCategory.TECHNICAL_EXCLUSION
            and not source_presence_complete(item)
        )
    ]

    total_event_count = sum(
        len(case.events)
        for case in anomaly_cases
    )
    multi_event_episode_ids = [
        case.episode_id
        for case in anomaly_cases
        if len(case.events) > 1
    ]

    payload = {
        "mode": (
            "day16_training_manifest_batch_audit"
        ),
        "source_manifest": str(
            manifest_path
        ),
        "source_encoding": (
            source["encoding"]
        ),
        "dataset_root": str(
            dataset_root
        ),
        "summary": summary,
        "expected_checks": (
            expected_checks
        ),
        "anomaly_ids_match_expected": (
            anomaly_ids_match
        ),
        "anomaly_review_case_count": len(
            anomaly_cases
        ),
        "anomaly_event_count": (
            total_event_count
        ),
        "multi_event_episode_ids": (
            multi_event_episode_ids
        ),
        "all_review_events_draft": all(
            case.all_events_draft
            for case in anomaly_cases
        ),
        "all_causal_diagnoses_unset": all(
            case.all_causal_diagnoses_unset
            for case in anomaly_cases
        ),
        "source_presence_complete": (
            all_source_presence_complete
        ),
        "required_anomaly_source_presence_complete": (
            anomaly_source_presence_complete
        ),
        "technical_exclusion_missing_sources": (
            technical_exclusion_missing_sources
        ),
        "audit_jsonl": (
            audit_path.relative_to(
                ROOT
            ).as_posix()
        ),
        "anomaly_review_jsonl": (
            review_path.relative_to(
                ROOT
            ).as_posix()
        ),
        "non_claims": [
            "original failure_reason is preserved but is not a causal diagnosis",
            "observed failure mode is an event label, not a causal root cause",
            "draft review events are not verified ground truth",
            "task_success and operation_anomaly are independent labels",
        ],
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
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
    print(
        "report="
        + report_path.relative_to(
            ROOT
        ).as_posix()
    )

    if not all(
        expected_checks.values()
    ):
        return 2
    if not anomaly_ids_match:
        return 3
    if len(anomaly_cases) != 8:
        return 4
    if not anomaly_source_presence_complete:
        return 5
    if total_event_count != 9:
        return 6
    if multi_event_episode_ids != [
        "20260815_111613"
    ]:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
