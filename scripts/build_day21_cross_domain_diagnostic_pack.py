from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from evidencemm.canonical_document_retriever import (
    CanonicalHybridDocumentRetriever,
)
from evidencemm.cross_domain_diagnostic_pack import (
    ARTIFACT_SCHEMA,
    CONFIG_SCHEMA,
    LOCALIZATION_ORIGIN,
    MANUAL_SUPPORT_STATUS,
    RobotIntervalEvidenceBuilder,
    build_artifact,
    build_case_bundle,
    extract_diagnostic_case_seeds,
    sha256_path,
    validate_manual_query_is_label_independent,
)


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
            "configs/day21_cross_domain_diagnostic_pack.yaml"
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
    if config.get(
        "schema_version"
    ) != CONFIG_SCHEMA:
        raise ValueError(
            "unexpected Day21 config schema_version"
        )

    inputs = config["inputs"]
    expected = config["expected"]

    day20_path = _resolve(
        project_root,
        inputs["day20_report_json"],
    )
    day19_model_path = _resolve(
        project_root,
        inputs["day19_model_json"],
    )
    document_manifest_path = _resolve(
        project_root,
        inputs["document_manifest"],
    )
    document_visual_manifest_path = _resolve(
        project_root,
        inputs["document_visual_manifest"],
    )

    day20_repository_path = str(
        inputs["day20_report_json"]
    )
    _assert_worktree_unchanged(
        project_root,
        day20_repository_path,
    )
    day20_blob_sha1 = _git_blob_sha1(
        project_root,
        day20_repository_path,
    )
    if day20_blob_sha1 != str(
        config["provenance"][
            "expected_day20_report_blob_sha1"
        ]
    ):
        raise ValueError(
            "Day20 report Git blob differs from "
            "the frozen Day21 expectation"
        )

    day20_report = json.loads(
        day20_path.read_text(
            encoding="utf-8"
        )
    )
    seeds = extract_diagnostic_case_seeds(
        day20_report,
        expected_event_ids=expected[
            "event_ids"
        ],
        expected_episode_ids=expected[
            "episode_ids"
        ],
    )
    if len(seeds) != int(
        expected["case_count"]
    ):
        raise ValueError(
            "Day21 case_count mismatch"
        )

    manual_query = str(
        config["manual_retrieval"][
            "query"
        ]
    )
    validate_manual_query_is_label_independent(
        manual_query
    )
    manual_top_k = int(
        config["manual_retrieval"][
            "top_k"
        ]
    )

    document_retriever = (
        CanonicalHybridDocumentRetriever(
            project_root=project_root,
            source_manifest_path=(
                inputs["document_manifest"]
            ),
            visual_manifest_path=(
                inputs[
                    "document_visual_manifest"
                ]
            ),
            hybrid_config_path=(
                inputs["hybrid_config"]
            ),
        )
    )
    manual_candidates = (
        document_retriever.search(
            manual_query,
            top_k=manual_top_k,
        )
    )
    manual_trace = (
        document_retriever.last_trace.to_dict()
        if document_retriever.last_trace
        is not None
        else None
    )
    document_retriever.release_models()

    if len(
        manual_candidates
    ) != manual_top_k:
        raise ValueError(
            "manual retriever returned unexpected top_k"
        )
    if not manual_candidates:
        raise ValueError(
            "manual retrieval produced no evidence"
        )
    if manual_candidates[
        0
    ].retriever_name != str(
        config["manual_retrieval"][
            "retriever"
        ]
    ):
        raise ValueError(
            "manual retriever name differs from "
            "Day21 canonical contract"
        )

    document_items = [
        candidate.item
        for candidate in manual_candidates
    ]

    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else _resolve(
            project_root,
            inputs["dataset_root"],
        ).resolve()
    )
    manifest_root = _resolve(
        project_root,
        inputs[
            "diagnostic_manifest_root"
        ],
    )
    processed_root = _resolve(
        project_root,
        inputs["processed_root"],
    )

    question = str(
        config["diagnostic_case"][
            "question"
        ]
    )
    case_records = []

    for seed in seeds:
        builder = RobotIntervalEvidenceBuilder(
            project_root=project_root,
            episode_manifest_path=(
                manifest_root
                / f"{seed.episode_id}.json"
            ),
            episode_dir=(
                dataset_root
                / seed.episode_id
            ),
            frame_records_path=(
                processed_root
                / seed.episode_id
                / "frames.jsonl"
            ),
        )

        (
            bundle,
            robot_items,
            readiness,
        ) = build_case_bundle(
            seed=seed,
            question=question,
            document_items=document_items,
            robot_builder=builder,
        )

        case_records.append(
            {
                "case_id": (
                    f"day21_{seed.event_id}"
                ),
                "event_id": seed.event_id,
                "episode_id": seed.episode_id,
                "localization_origin": (
                    LOCALIZATION_ORIGIN
                ),
                "localized_proposal": {
                    "start_frame": (
                        seed.proposal_start_frame
                    ),
                    "center_frame": (
                        seed.proposal_center_frame
                    ),
                    "end_frame": (
                        seed.proposal_end_frame
                    ),
                    "representative_frames": [
                        int(
                            item.payload.frame_index
                        )
                        for item in robot_items
                    ],
                },
                "manual_support_status": (
                    MANUAL_SUPPORT_STATUS
                ),
                "evidence_bundle": (
                    bundle.model_dump(
                        mode="json"
                    )
                ),
                "readiness": readiness,
            }
        )

    artifact = build_artifact(
        frozen_after_day20_commit=str(
            config["provenance"][
                "frozen_after_day20_commit"
            ]
        ),
        day20_report_blob_sha1=(
            day20_blob_sha1
        ),
        day20_report_sha256=(
            sha256_path(
                day20_path
            )
        ),
        day19_model_sha256=(
            sha256_path(
                day19_model_path
            )
        ),
        document_manifest_sha256=(
            sha256_path(
                document_manifest_path
            )
        ),
        document_visual_manifest_sha256=(
            sha256_path(
                document_visual_manifest_path
            )
        ),
        manual_query=manual_query,
        manual_retriever_name=(
            manual_candidates[
                0
            ].retriever_name
        ),
        manual_top_k=manual_top_k,
        manual_candidates=(
            manual_candidates
        ),
        manual_trace=manual_trace,
        case_records=case_records,
    )

    if artifact[
        "schema_version"
    ] != ARTIFACT_SCHEMA:
        raise ValueError(
            "Day21 artifact schema mismatch"
        )

    if artifact["summary"][
        "root_cause_answerable_count"
    ] != int(
        expected[
            "root_cause_answerable_count"
        ]
    ):
        raise ValueError(
            "Day21 root_cause_answerable_count mismatch"
        )
    if artifact["summary"][
        "abstain_count"
    ] != int(
        expected["abstain_count"]
    ):
        raise ValueError(
            "Day21 abstain_count mismatch"
        )

    output_path = _resolve(
        project_root,
        config["output"][
            "artifact_json"
        ],
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
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
                "artifact_status": (
                    artifact[
                        "artifact_status"
                    ]
                ),
                "case_count": (
                    artifact["summary"][
                        "case_count"
                    ]
                ),
                "manual_retriever_name": (
                    artifact[
                        "manual_retrieval"
                    ]["retriever_name"]
                ),
                "manual_page_numbers": [
                    item["page_number"]
                    for item in artifact[
                        "manual_retrieval"
                    ]["candidates"]
                ],
                "document_items_per_case": (
                    int(
                        expected[
                            "document_items_per_case"
                        ]
                    )
                ),
                "robot_items_per_case": (
                    int(
                        expected[
                            "robot_items_per_case"
                        ]
                    )
                ),
                "root_cause_answerable_count": (
                    artifact["summary"][
                        "root_cause_answerable_count"
                    ]
                ),
                "abstain_count": (
                    artifact["summary"][
                        "abstain_count"
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
