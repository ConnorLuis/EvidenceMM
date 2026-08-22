from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.robot_failure_dataset import (
    AuditCategory,
    ReviewStatus,
    load_anomaly_review_cases,
    load_source_audit,
    summarize_source_audit,
)
from evidencemm.temporal_evidence import (
    EpisodeManifest,
    load_frame_records,
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
        "--require-bound-anomalies",
        action="store_true",
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

    audit_path = (
        ROOT
        / config["audit"][
            "output_jsonl"
        ]
    )
    review_path = (
        ROOT
        / config["audit"][
            "anomaly_review_jsonl"
        ]
    )

    audit = load_source_audit(
        audit_path
    )
    reviews = load_anomaly_review_cases(
        review_path
    )

    summary = summarize_source_audit(
        audit
    )

    expected = config[
        "expected_current_source_audit"
    ]
    categories = summary[
        "categories"
    ]

    checks = {
        "total_rows": (
            summary["total_rows"]
            == int(expected["total_rows"])
        ),
        "clean_reference_candidate": (
            categories.get(
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
            categories.get(
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
            categories.get(
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
            categories.get(
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
            categories.get(
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

    anomaly_audit = [
        item
        for item in audit
        if (
            item.audit_category
            == AuditCategory.OPERATION_ANOMALY
        )
    ]

    audit_by_id = {
        item.episode_id: item
        for item in anomaly_audit
    }
    review_by_id = {
        item.episode_id: item
        for item in reviews
    }

    event_count = sum(
        len(review.events)
        for review in reviews
    )
    multi_event_ids = [
        review.episode_id
        for review in reviews
        if len(review.events) > 1
    ]

    review_integrity = {
        "review_count_is_8": (
            len(reviews) == 8
        ),
        "event_count_is_9": (
            event_count == 9
        ),
        "multi_event_episode_is_111613": (
            multi_event_ids
            == ["20260815_111613"]
        ),
        "review_ids_match_audit": (
            set(review_by_id)
            == set(audit_by_id)
        ),
        "all_events_draft": all(
            event.event_status
            == ReviewStatus.DRAFT
            for review in reviews
            for event in review.events
        ),
        "all_causal_diagnoses_unset": all(
            event.causal_diagnosis
            is None
            for review in reviews
            for event in review.events
        ),
        "task_success_preserved": all(
            review.task_success
            == audit_by_id[
                review.episode_id
            ].task_success
            for review in reviews
            if review.episode_id
            in audit_by_id
        ),
        "failure_reason_preserved": all(
            review.original_failure_reason
            == audit_by_id[
                review.episode_id
            ].original_failure_reason
            for review in reviews
            if review.episode_id
            in audit_by_id
        ),
        "observed_modes_preserved": all(
            review.observed_failure_modes
            == audit_by_id[
                review.episode_id
            ].observed_failure_modes
            for review in reviews
            if review.episode_id
            in audit_by_id
        ),
    }

    bound_errors = []
    bound_count = 0

    binding = config[
        "diagnostic_binding"
    ]

    for review in reviews:
        manifest_path = (
            ROOT
            / binding["manifest_root"]
            / f"{review.episode_id}.json"
        )
        frames_path = (
            ROOT
            / binding["processed_root"]
            / review.episode_id
            / "frames.jsonl"
        )

        if not (
            manifest_path.is_file()
            and frames_path.is_file()
        ):
            if args.require_bound_anomalies:
                bound_errors.append(
                    {
                        "episode_id": (
                            review.episode_id
                        ),
                        "error": (
                            "diagnostic manifest or frames.jsonl missing"
                        ),
                    }
                )
            continue

        try:
            manifest = (
                EpisodeManifest.model_validate_json(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
            records = load_frame_records(
                frames_path
            )
            if (
                manifest.episode_id
                != review.episode_id
            ):
                raise ValueError(
                    "manifest episode_id mismatch"
                )
            frame_ids = {
                item.frame_index
                for item in records
            }
            if frame_ids != set(
                range(
                    manifest.frame_count
                )
            ):
                raise ValueError(
                    "diagnostic frame records do not cover episode"
                )
            bound_count += 1
        except Exception as exc:
            bound_errors.append(
                {
                    "episode_id": (
                        review.episode_id
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    payload = {
        "mode": (
            "day16_failure_data_layer_validation"
        ),
        "audit_checks": checks,
        "review_integrity": (
            review_integrity
        ),
        "review_case_count": (
            len(reviews)
        ),
        "review_event_count": (
            event_count
        ),
        "multi_event_episode_ids": (
            multi_event_ids
        ),
        "bound_anomaly_count": (
            bound_count
        ),
        "bound_errors": (
            bound_errors
        ),
        "source_summary": (
            summary
        ),
        "data_layer_valid": (
            all(checks.values())
            and all(
                review_integrity.values()
            )
            and not bound_errors
            and (
                not args.require_bound_anomalies
                or bound_count == 8
            )
        ),
        "manual_review_complete": all(
            review.manual_review_complete
            for review in reviews
        ),
        "non_claims": [
            "data_layer_valid does not mean causal diagnosis is verified",
            "the nine imported anomaly events are draft until human review",
            "task success and operation anomaly remain separate labels",
        ],
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not payload[
        "data_layer_valid"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
