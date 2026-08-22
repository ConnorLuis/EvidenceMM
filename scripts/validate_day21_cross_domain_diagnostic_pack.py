from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from evidencemm.cross_domain_diagnostic_pack import (
    ARTIFACT_SCHEMA,
    ARTIFACT_STATUS,
    CONFIG_SCHEMA,
    RobotIntervalEvidenceBuilder,
    assert_no_review_labels_leaked,
    canonical_json_bytes,
    extract_diagnostic_case_seeds,
    representative_frames,
    sha256_path,
    validate_case_record_shape,
    validate_manual_query_is_label_independent,
)
from evidencemm.document_candidate_retrieval import (
    DocumentBM25CandidateRetriever,
)
from evidencemm.unified_evidence import (
    UnifiedEvidenceBundle,
    validate_cross_domain_bundle,
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
        "--artifact",
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

    artifact_path = (
        args.artifact
        if args.artifact is not None
        else _resolve(
            project_root,
            config["output"][
                "artifact_json"
            ],
        )
    )
    if not artifact_path.is_absolute():
        artifact_path = (
            project_root
            / artifact_path
        )
    artifact = json.loads(
        artifact_path.read_text(
            encoding="utf-8"
        )
    )

    if artifact.get(
        "schema_version"
    ) != ARTIFACT_SCHEMA:
        raise ValueError(
            "unexpected Day21 artifact schema_version"
        )
    if artifact.get(
        "artifact_status"
    ) != ARTIFACT_STATUS:
        raise ValueError(
            "unexpected Day21 artifact_status"
        )

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
        inputs[
            "document_visual_manifest"
        ],
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
            "Day20 blob differs from Day21 frozen contract"
        )

    expected_provenance = {
        "frozen_after_day20_commit": str(
            config["provenance"][
                "frozen_after_day20_commit"
            ]
        ),
        "day20_report_blob_sha1": (
            day20_blob_sha1
        ),
        "day20_report_sha256": (
            sha256_path(
                day20_path
            )
        ),
        "day19_model_sha256": (
            sha256_path(
                day19_model_path
            )
        ),
        "document_manifest_sha256": (
            sha256_path(
                document_manifest_path
            )
        ),
        "document_visual_manifest_sha256": (
            sha256_path(
                document_visual_manifest_path
            )
        ),
    }
    if artifact.get(
        "provenance"
    ) != expected_provenance:
        raise ValueError(
            "Day21 provenance mismatch"
        )

    manual_query = str(
        config["manual_retrieval"][
            "query"
        ]
    )
    validate_manual_query_is_label_independent(
        manual_query
    )
    if artifact[
        "manual_retrieval"
    ]["query"] != manual_query:
        raise ValueError(
            "Day21 manual query mismatch"
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
    seed_by_event = {
        seed.event_id: seed
        for seed in seeds
    }

    cases = artifact.get(
        "cases",
        [],
    )
    if len(cases) != int(
        expected["case_count"]
    ):
        raise ValueError(
            "Day21 artifact case_count mismatch"
        )

    document_base = (
        DocumentBM25CandidateRetriever(
            project_root=project_root,
            source_manifest_path=(
                inputs["document_manifest"]
            ),
            visual_manifest_path=(
                inputs[
                    "document_visual_manifest"
                ]
            ),
        )
    )
    canonical_document_items = {
        candidate.item.payload.page_number: (
            candidate.item
        )
        for candidate in document_base.search(
            manual_query,
            top_k=len(
                document_base.documents
            ),
        )
    }

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

    seen_events = set()

    for case in cases:
        event_id = str(
            case["event_id"]
        )
        if event_id in seen_events:
            raise ValueError(
                "duplicate Day21 case event_id"
            )
        seen_events.add(
            event_id
        )
        if event_id not in seed_by_event:
            raise ValueError(
                "Day21 case event not found in Day20"
            )

        seed = seed_by_event[
            event_id
        ]
        if case["episode_id"] != (
            seed.episode_id
        ):
            raise ValueError(
                "Day21 case episode mismatch"
            )

        validate_case_record_shape(
            case,
            expected_document_items=int(
                expected[
                    "document_items_per_case"
                ]
            ),
            expected_robot_items=int(
                expected[
                    "robot_items_per_case"
                ]
            ),
        )

        localized = case[
            "localized_proposal"
        ]
        expected_localized = {
            "start_frame": (
                seed.proposal_start_frame
            ),
            "center_frame": (
                seed.proposal_center_frame
            ),
            "end_frame": (
                seed.proposal_end_frame
            ),
            "representative_frames": list(
                representative_frames(
                    seed
                )
            ),
        }
        if localized != (
            expected_localized
        ):
            raise ValueError(
                "Day21 localized proposal mismatch"
            )

        bundle = (
            UnifiedEvidenceBundle.model_validate(
                case[
                    "evidence_bundle"
                ]
            )
        )
        valid, errors = (
            validate_cross_domain_bundle(
                bundle
            )
        )
        if not valid:
            raise ValueError(
                "invalid Day21 evidence bundle: "
                + repr(errors)
            )

        document_items = [
            item
            for item in bundle.items
            if item.kind.value
            == "document_page"
        ]
        robot_items = [
            item
            for item in bundle.items
            if item.kind.value
            == "robot_sample"
        ]

        for item in document_items:
            page = int(
                item.payload.page_number
            )
            expected_item = (
                canonical_document_items[
                    page
                ]
            )
            if canonical_json_bytes(
                item.model_dump(
                    mode="json"
                )
            ) != canonical_json_bytes(
                expected_item.model_dump(
                    mode="json"
                )
            ):
                raise ValueError(
                    f"document page {page} differs from "
                    "canonical source evidence"
                )

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
        expected_robot_by_frame = {
            frame: builder.build_item(
                frame
            )
            for frame in representative_frames(
                seed
            )
        }

        for item in robot_items:
            frame = int(
                item.payload.frame_index
            )
            if frame not in expected_robot_by_frame:
                raise ValueError(
                    "robot evidence frame outside localized proposal samples"
                )
            expected_item = (
                expected_robot_by_frame[
                    frame
                ]
            )
            if canonical_json_bytes(
                item.model_dump(
                    mode="json"
                )
            ) != canonical_json_bytes(
                expected_item.model_dump(
                    mode="json"
                )
            ):
                raise ValueError(
                    f"robot frame {frame} differs from "
                    "canonical source evidence"
                )

    if seen_events != set(
        str(value)
        for value in expected[
            "event_ids"
        ]
    ):
        raise ValueError(
            "Day21 validated event universe mismatch"
        )

    assert_no_review_labels_leaked(
        artifact
    )

    summary = artifact.get(
        "summary",
        {},
    )
    if summary.get(
        "root_cause_answerable_count"
    ) != int(
        expected[
            "root_cause_answerable_count"
        ]
    ):
        raise ValueError(
            "root_cause_answerable_count mismatch"
        )
    if summary.get(
        "abstain_count"
    ) != int(
        expected["abstain_count"]
    ):
        raise ValueError(
            "abstain_count mismatch"
        )

    print(
        json.dumps(
            {
                "valid": True,
                "artifact_status": (
                    artifact[
                        "artifact_status"
                    ]
                ),
                "case_count": (
                    summary[
                        "case_count"
                    ]
                ),
                "manual_retriever_name": (
                    artifact[
                        "manual_retrieval"
                    ][
                        "retriever_name"
                    ]
                ),
                "manual_page_numbers": [
                    row[
                        "page_number"
                    ]
                    for row in artifact[
                        "manual_retrieval"
                    ]["candidates"]
                ],
                "root_cause_answerable_count": (
                    summary[
                        "root_cause_answerable_count"
                    ]
                ),
                "abstain_count": (
                    summary[
                        "abstain_count"
                    ]
                ),
                "review_label_fields_leaked": False,
                "physical_root_cause_claimed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
