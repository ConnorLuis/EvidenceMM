from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from evidencemm.root_cause_benchmark_v2 import (
    CONFIG_SCHEMA,
    build_collection_plan,
    build_protocol_artifact,
    canonical_json_bytes,
    collection_plan_csv_bytes,
    collection_plan_summary,
    load_collection_plan_csv,
    sha256_bytes,
    validate_protocol_artifact,
)


FROZEN_REPO_PATHS = {
    "task_definition": "docs/task_definition.md",
    "day21_doc": "docs/day21_cross_domain_diagnostic_pack.md",
    "day21_artifact": "data/eval/day21_cross_domain_diagnostic_cases.json",
}


def _resolve(
    root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _git_blob_sha1(
    root: Path,
    repository_path: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            f"HEAD:{repository_path}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _assert_unmodified(
    root: Path,
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
        cwd=root,
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
            "configs/day22_root_cause_benchmark_v2.yaml"
        ),
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    config_path = _resolve(
        root,
        args.config,
    )
    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )
    if config.get(
        "schema_version"
    ) != CONFIG_SCHEMA:
        raise ValueError(
            "unexpected Day22 config schema_version"
        )

    for path in FROZEN_REPO_PATHS.values():
        _assert_unmodified(
            root,
            path,
        )

    actual_blobs = {
        name: _git_blob_sha1(
            root,
            path,
        )
        for name, path in (
            FROZEN_REPO_PATHS.items()
        )
    }
    expected_blobs = {
        "task_definition": str(
            config["provenance"][
                "expected_task_definition_blob_sha1"
            ]
        ),
        "day21_doc": str(
            config["provenance"][
                "expected_day21_doc_blob_sha1"
            ]
        ),
        "day21_artifact": str(
            config["provenance"][
                "expected_day21_artifact_blob_sha1"
            ]
        ),
    }
    if actual_blobs != expected_blobs:
        raise ValueError(
            "frozen roadmap provenance mismatch"
        )

    plan_path = _resolve(
        root,
        config["outputs"][
            "collection_plan_csv"
        ],
    )
    protocol_path = _resolve(
        root,
        config["outputs"][
            "protocol_json"
        ],
    )

    plan_bytes = plan_path.read_bytes()
    loaded_rows = load_collection_plan_csv(
        plan_bytes.decode(
            "utf-8"
        )
    )
    rebuilt_rows = build_collection_plan()
    rebuilt_plan_bytes = (
        collection_plan_csv_bytes(
            rebuilt_rows
        )
    )

    if plan_bytes != rebuilt_plan_bytes:
        raise ValueError(
            "Day22 collection plan differs from deterministic rebuild"
        )

    plan_sha256 = sha256_bytes(
        plan_bytes
    )

    loaded_artifact = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )
    validate_protocol_artifact(
        loaded_artifact,
        expected_plan_sha256=(
            plan_sha256
        ),
    )

    rebuilt_artifact = build_protocol_artifact(
        config=config,
        collection_plan_sha256=(
            plan_sha256
        ),
        frozen_blob_sha1=(
            actual_blobs
        ),
    )

    if canonical_json_bytes(
        loaded_artifact
    ) != canonical_json_bytes(
        rebuilt_artifact
    ):
        raise ValueError(
            "Day22 protocol artifact differs from deterministic rebuild"
        )

    summary = collection_plan_summary(
        loaded_rows
    )
    future_split = loaded_artifact[
        "future_split"
    ]
    leakage = loaded_artifact[
        "anti_label_leakage"
    ]

    print(
        json.dumps(
            {
                "valid": True,
                "protocol_status": (
                    loaded_artifact[
                        "protocol_status"
                    ]
                ),
                "target_episode_count": (
                    summary[
                        "target_episode_count"
                    ]
                ),
                "pair_group_count": (
                    summary[
                        "pair_group_count"
                    ]
                ),
                "controlled_cause_counts": (
                    summary[
                        "controlled_cause_counts"
                    ]
                ),
                "clean_control_count": (
                    summary[
                        "role_counts"
                    ][
                        "clean_control"
                    ]
                ),
                "insufficient_evidence_candidate_count": (
                    summary[
                        "role_counts"
                    ][
                        "insufficient_evidence_candidate"
                    ]
                ),
                "pilot_episode_count_excluded": (
                    int(
                        loaded_artifact[
                            "pilot"
                        ][
                            "expected_episode_count"
                        ]
                    )
                ),
                "future_split_membership_materialized": (
                    future_split[
                        "materialize_membership_on_day22"
                    ]
                ),
                "held_out_model_selection_allowed": (
                    future_split[
                        "held_out_model_selection_allowed"
                    ]
                ),
                "admin_labels_visible_to_model": (
                    False
                ),
                "source_manifest_embeds_admin_labels": (
                    not bool(
                        leakage[
                            "source_manifest_must_not_embed_admin_labels"
                        ]
                    )
                ),
                "numeric_intervention_values_frozen_on_day22": (
                    loaded_artifact[
                        "acceptance"
                    ][
                        "day22_freezes_numeric_intervention_values"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
