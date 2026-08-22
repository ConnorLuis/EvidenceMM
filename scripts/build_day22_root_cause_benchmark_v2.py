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
    collection_plan_csv_bytes,
    collection_plan_summary,
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
            "Day22 frozen roadmap provenance differs from expected: "
            f"expected={expected_blobs}, actual={actual_blobs}"
        )

    rows = build_collection_plan()
    csv_bytes = collection_plan_csv_bytes(
        rows
    )
    plan_sha256 = sha256_bytes(
        csv_bytes
    )

    artifact = build_protocol_artifact(
        config=config,
        collection_plan_sha256=(
            plan_sha256
        ),
        frozen_blob_sha1=(
            actual_blobs
        ),
    )
    validate_protocol_artifact(
        artifact,
        expected_plan_sha256=(
            plan_sha256
        ),
    )

    plan_path = _resolve(
        root,
        config["outputs"][
            "collection_plan_csv"
        ],
    )
    plan_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plan_path.write_bytes(
        csv_bytes
    )

    protocol_path = _resolve(
        root,
        config["outputs"][
            "protocol_json"
        ],
    )
    protocol_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    protocol_path.write_text(
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
            {
                "schema_version": (
                    artifact[
                        "schema_version"
                    ]
                ),
                "protocol_status": (
                    artifact[
                        "protocol_status"
                    ]
                ),
                "collection_plan_sha256": (
                    plan_sha256
                ),
                "collection_plan": (
                    collection_plan_summary(
                        rows
                    )
                ),
                "pilot_episode_count": (
                    int(
                        config[
                            "pilot"
                        ][
                            "expected_episode_count"
                        ]
                    )
                ),
                "pilot_final_benchmark_eligible": (
                    False
                ),
                "future_split_membership_materialized": (
                    False
                ),
                "next_day": (
                    config[
                        "roadmap"
                    ]["day23"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
