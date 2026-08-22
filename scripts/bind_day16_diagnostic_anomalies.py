from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.diagnostic_episode_binding import (
    bind_diagnostic_episode,
    diagnostic_binding_allowed,
)
from evidencemm.robot_failure_dataset import (
    AuditCategory,
    load_source_audit,
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
    records = load_source_audit(
        audit_path
    )

    anomalies = [
        item
        for item in records
        if (
            item.audit_category
            == AuditCategory.OPERATION_ANOMALY
        )
    ]

    binding = config[
        "diagnostic_binding"
    ]

    results = []
    errors = []

    for item in anomalies:
        if not diagnostic_binding_allowed(
            item
        ):
            errors.append(
                {
                    "episode_id": (
                        item.episode_id
                    ),
                    "error": (
                        "audit record is not allowed "
                        "for diagnostic binding"
                    ),
                }
            )
            continue

        try:
            result = bind_diagnostic_episode(
                episode_dir=(
                    item.raw_episode_dir
                ),
                manifest_path=(
                    ROOT
                    / binding["manifest_root"]
                    / f"{item.episode_id}.json"
                ),
                frames_path=(
                    ROOT
                    / binding["processed_root"]
                    / item.episode_id
                    / "frames.jsonl"
                ),
                timestamp_source=(
                    binding[
                        "timestamp_source"
                    ]
                ),
            )
            results.append(
                {
                    "episode_id": (
                        result.episode_id
                    ),
                    "frame_count": (
                        result.frame_count
                    ),
                    "episode_sha256": (
                        result.episode_sha256
                    ),
                    "source_status": (
                        result.source_status
                    ),
                    "source_valid_for_training": (
                        result.source_valid_for_training
                    ),
                    "source_overall_pass": (
                        result.source_overall_pass
                    ),
                    "source_failed_checks": list(
                        result.source_failed_checks
                    ),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "episode_id": (
                        item.episode_id
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    payload = {
        "mode": (
            "day16_batch_diagnostic_binding"
        ),
        "requested_anomaly_count": len(
            anomalies
        ),
        "bound_count": len(
            results
        ),
        "error_count": len(
            errors
        ),
        "results": results,
        "errors": errors,
        "training_acceptance_required": False,
        "technical_exclusions_bound": False,
        "non_claims": [
            "valid_for_training is preserved but is not a diagnostic acceptance gate",
            "overall_pass is preserved but is not a diagnostic acceptance gate",
            "technical exclusions remain excluded by source audit",
        ],
    }

    report_path = (
        ROOT
        / binding[
            "report_json"
        ]
    )
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

    if errors:
        return 2
    if len(results) != 8:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
