from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CONFIG_SCHEMA = (
    "evidencemm_day21_cross_domain_diagnostic_pack_config_v1"
)
ARTIFACT_SCHEMA = (
    "evidencemm_day21_cross_domain_diagnostic_cases_v1"
)
ARTIFACT_STATUS = (
    "post_heldout_cross_domain_diagnostic_evidence_pack"
)
DAY20_REPORT_SCHEMA = (
    "evidencemm_day20_heldout_interval_eval_report_v1"
)
DAY20_STATUS = (
    "prospective_procedural_heldout_interval_evaluation"
)

LOCALIZATION_ORIGIN = (
    "day20_gt_matched_best_proposal_for_post_eval_diagnostics"
)
MANUAL_SUPPORT_STATUS = (
    "retrieved_candidates_unlabeled_for_causal_support"
)
ROOT_CAUSE_DECISION = "abstain_physical_root_cause"

FORBIDDEN_LABEL_FIELDS = {
    "gold_start_frame",
    "gold_end_frame",
    "observed_failure_mode",
    "best_iou",
    "onset_abs_error_frames",
    "offset_abs_error_frames",
}


@dataclass(frozen=True)
class DiagnosticCaseSeed:
    event_id: str
    episode_id: str
    proposal_start_frame: int
    proposal_center_frame: int
    proposal_end_frame: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError(
            f"{path}: expected JSON object"
        )
    return value


def validate_day20_report_contract(
    report: dict[str, Any],
) -> None:
    if report.get(
        "schema_version"
    ) != DAY20_REPORT_SCHEMA:
        raise ValueError(
            "unexpected Day20 report schema_version"
        )
    if report.get(
        "evaluation_status"
    ) != DAY20_STATUS:
        raise ValueError(
            "unexpected Day20 evaluation_status"
        )
    if report.get(
        "evaluation_split"
    ) != "held_out":
        raise ValueError(
            "Day20 report must be held_out"
        )

    anti = report.get(
        "anti_leakage",
        {},
    )
    for key in (
        "model_selection_performed",
        "radius_tuned_on_held_out",
        "post_heldout_tuning_allowed",
    ):
        if anti.get(key) is not False:
            raise ValueError(
                f"Day20 frozen evaluation contract failed: {key}"
            )

    seal = report.get(
        "evaluation_seal",
        {},
    )
    if seal.get(
        "frozen_model_final_evaluation"
    ) is not True:
        raise ValueError(
            "Day20 report is not sealed as frozen final evaluation"
        )
    if seal.get(
        "same_held_out_set_may_be_used_for_future_model_selection"
    ) is not False:
        raise ValueError(
            "Day20 held-out set is not sealed against model selection"
        )


def _proposal_key(
    proposal: dict[str, Any],
) -> tuple[int, int, int]:
    return (
        int(proposal["start_frame"]),
        int(proposal["center_frame"]),
        int(proposal["end_frame"]),
    )


def extract_diagnostic_case_seeds(
    report: dict[str, Any],
    *,
    expected_event_ids: Sequence[str],
    expected_episode_ids: Sequence[str],
) -> list[DiagnosticCaseSeed]:
    validate_day20_report_contract(
        report
    )

    expected_events = sorted(
        str(value)
        for value in expected_event_ids
    )
    expected_episodes = sorted(
        str(value)
        for value in expected_episode_ids
    )

    results = report.get(
        "event_results",
        [],
    )
    if not isinstance(
        results,
        list,
    ):
        raise ValueError(
            "Day20 event_results must be a list"
        )

    proposals_by_episode = report.get(
        "held_out_proposals",
        {},
    )

    seeds: list[DiagnosticCaseSeed] = []
    seen_events: set[str] = set()

    for row in results:
        event_id = str(
            row["event_id"]
        )
        episode_id = str(
            row["episode_id"]
        )
        if event_id in seen_events:
            raise ValueError(
                f"duplicate Day20 event result: {event_id}"
            )
        seen_events.add(event_id)

        proposal = {
            "start_frame": int(
                row["best_proposal_start_frame"]
            ),
            "center_frame": int(
                row["best_proposal_center_frame"]
            ),
            "end_frame": int(
                row["best_proposal_end_frame"]
            ),
        }
        key = _proposal_key(
            proposal
        )

        episode_proposals = proposals_by_episode.get(
            episode_id,
            [],
        )
        valid_keys = {
            _proposal_key(item)
            for item in episode_proposals
        }
        if key not in valid_keys:
            raise ValueError(
                f"{event_id}: Day20 best proposal is not "
                "present in frozen held_out_proposals"
            )

        if not (
            proposal["start_frame"]
            <= proposal["center_frame"]
            <= proposal["end_frame"]
        ):
            raise ValueError(
                f"{event_id}: invalid proposal frame order"
            )

        seeds.append(
            DiagnosticCaseSeed(
                event_id=event_id,
                episode_id=episode_id,
                proposal_start_frame=(
                    proposal["start_frame"]
                ),
                proposal_center_frame=(
                    proposal["center_frame"]
                ),
                proposal_end_frame=(
                    proposal["end_frame"]
                ),
            )
        )

    seeds.sort(
        key=lambda item: item.event_id
    )

    actual_events = [
        seed.event_id
        for seed in seeds
    ]
    actual_episodes = sorted(
        seed.episode_id
        for seed in seeds
    )

    if actual_events != expected_events:
        raise ValueError(
            f"Day21 event universe mismatch: "
            f"expected={expected_events}, actual={actual_events}"
        )
    if actual_episodes != expected_episodes:
        raise ValueError(
            f"Day21 episode universe mismatch: "
            f"expected={expected_episodes}, actual={actual_episodes}"
        )

    return seeds


def representative_frames(
    seed: DiagnosticCaseSeed,
) -> tuple[int, ...]:
    values = (
        seed.proposal_start_frame,
        seed.proposal_center_frame,
        seed.proposal_end_frame,
    )
    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError(
            "diagnostic case requires representative frames"
        )
    return tuple(result)


def validate_manual_query_is_label_independent(
    query: str,
) -> None:
    normalized = query.strip().lower()
    if not normalized:
        raise ValueError(
            "manual retrieval query must be non-empty"
        )

    forbidden = (
        "grasp_drop",
        "post_place_collision",
        "drop_above_target",
        "object_push_during_grasp",
        "target_offset_or_perception",
        "gripper_close_timing",
        "trajectory_execution_deviation",
    )
    leaking = [
        term
        for term in forbidden
        if term in normalized
    ]
    if leaking:
        raise ValueError(
            "manual query leaks reviewed failure labels: "
            + ", ".join(leaking)
        )


def build_readiness(
    *,
    cross_domain_bundle_valid: bool,
    document_item_count: int,
    robot_item_count: int,
    localization_origin: str,
    manual_support_status: str,
) -> dict[str, Any]:
    if localization_origin != LOCALIZATION_ORIGIN:
        raise ValueError(
            "unexpected Day21 localization_origin"
        )
    if manual_support_status != MANUAL_SUPPORT_STATUS:
        raise ValueError(
            "unexpected Day21 manual_support_status"
        )

    root_cause_answerable = False

    return {
        "decision": ROOT_CAUSE_DECISION,
        "root_cause_answerable": root_cause_answerable,
        "checks": {
            "cross_domain_bundle_valid": bool(
                cross_domain_bundle_valid
            ),
            "document_evidence_available": (
                int(document_item_count) > 0
            ),
            "robot_visual_state_action_evidence_available": (
                int(robot_item_count) > 0
            ),
            "manual_causal_ground_truth_available": False,
            "causal_supervision_available": False,
            "end_to_end_localization_inference_claim": False,
        },
        "uncertainty_reasons": [
            (
                "day20_best_proposal_is_gt_matched_for_post_eval_"
                "diagnostic_analysis_not_end_to_end_inference"
            ),
            (
                "retrieved_manual_pages_are_candidates_without_"
                "human_verified_causal_support_labels"
            ),
            (
                "current_verified_robot_events_do_not_provide_"
                "discriminative_physical_cause_supervision"
            ),
        ],
    }


def assert_no_review_labels_leaked(
    value: Any,
    *,
    path: str = "$",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_LABEL_FIELDS:
                raise ValueError(
                    f"Day21 artifact leaks Day20 review label "
                    f"field {path}.{key}"
                )
            assert_no_review_labels_leaked(
                child,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_review_labels_leaked(
                child,
                path=f"{path}[{index}]",
            )


def _stored_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(
            project_root.resolve()
        ).as_posix()
    except ValueError:
        return str(resolved)


class RobotIntervalEvidenceBuilder:
    def __init__(
        self,
        *,
        project_root: str | Path,
        episode_manifest_path: str | Path,
        episode_dir: str | Path,
        frame_records_path: str | Path,
    ) -> None:
        from evidencemm.robot_candidate_retrieval import (
            RobotSignalCandidateRetriever,
        )

        self.project_root = Path(
            project_root
        ).resolve()
        self.retriever = (
            RobotSignalCandidateRetriever(
                project_root=self.project_root,
                episode_manifest_path=(
                    episode_manifest_path
                ),
                episode_dir=episode_dir,
                frame_records_path=frame_records_path,
            )
        )

    @property
    def episode_id(self) -> str:
        return str(
            self.retriever.manifest.episode_id
        )

    def build_item(
        self,
        frame_index: int,
    ):
        from evidencemm.data_binding import (
            sha256_file,
        )
        from evidencemm.schemas import (
            EvidenceRef,
            SourceType,
        )
        from evidencemm.unified_evidence import (
            EvidenceProvenance,
            RobotCameraAsset,
            RobotSamplePayload,
            RobotStateActionSnapshot,
            UnifiedEvidenceItem,
            UnifiedEvidenceKind,
        )

        frame = int(frame_index)
        if frame < 0 or frame >= len(
            self.retriever.samples
        ):
            raise ValueError(
                f"frame outside episode: {frame}"
            )

        sample = self.retriever.samples[
            frame
        ]
        front, wrist = (
            self.retriever.pairs[
                frame
            ]
        )
        pair = [front, wrist]

        if sample.frame_index != frame:
            raise ValueError(
                "state/action frame index mismatch"
            )

        for record in pair:
            image_path = (
                self.retriever.episode_dir
                / record.image_relpath
            )
            if not image_path.is_file():
                raise FileNotFoundError(
                    image_path
                )
            if sha256_file(
                image_path
            ) != record.image_sha256:
                raise ValueError(
                    f"{record.camera} image SHA256 mismatch"
                )

        refs = [
            EvidenceRef(
                source_id=self.episode_id,
                source_type=(
                    SourceType.ROBOT_SEQUENCE
                ),
                time_start_sec=(
                    sample.timestamp_sec
                ),
                time_end_sec=(
                    sample.timestamp_sec
                ),
                frame_index=frame,
                camera=record.camera,
            )
            for record in pair
        ]

        return UnifiedEvidenceItem(
            evidence_id=(
                f"robot:{self.episode_id}:f{frame}"
            ),
            kind=(
                UnifiedEvidenceKind.ROBOT_SAMPLE
            ),
            refs=refs,
            provenance=EvidenceProvenance(
                source_id=self.episode_id,
                source_type=(
                    SourceType.ROBOT_SEQUENCE
                ),
                manifest_path=_stored_path(
                    self.retriever.episode_manifest_path,
                    project_root=self.project_root,
                ),
                canonical_sha256=(
                    self.retriever.manifest.episode_sha256
                ),
                supporting_sha256={
                    "metadata.json": (
                        self.retriever.manifest.metadata_sha256
                    ),
                    "samples.csv": (
                        self.retriever.manifest.samples_csv_sha256
                    ),
                },
            ),
            payload=RobotSamplePayload(
                episode_id=self.episode_id,
                frame_index=frame,
                timestamp_sec=(
                    sample.timestamp_sec
                ),
                cameras=[
                    RobotCameraAsset(
                        camera=record.camera,
                        frame_index=record.frame_index,
                        timestamp_sec=record.timestamp_sec,
                        image_relpath=record.image_relpath,
                        image_sha256=record.image_sha256,
                        source_timestamp_ns=(
                            record.source_timestamp_ns
                        ),
                        source_age_ms=(
                            record.source_age_ms
                        ),
                        width_px=record.width_px,
                        height_px=record.height_px,
                    )
                    for record in pair
                ],
                state_action=(
                    RobotStateActionSnapshot(
                        frame_index=sample.frame_index,
                        timestamp_sec=(
                            sample.timestamp_sec
                        ),
                        observation=(
                            sample.observation
                        ),
                        action=sample.action,
                        tracking_error=(
                            sample.tracking_error
                        ),
                    )
                ),
            ),
        )


def build_case_bundle(
    *,
    seed: DiagnosticCaseSeed,
    question: str,
    document_items: Sequence[Any],
    robot_builder: RobotIntervalEvidenceBuilder,
):
    from evidencemm.unified_evidence import (
        UnifiedEvidenceBundle,
        validate_cross_domain_bundle,
    )

    if robot_builder.episode_id != (
        seed.episode_id
    ):
        raise ValueError(
            "robot evidence builder episode differs from case"
        )

    robot_items = [
        robot_builder.build_item(
            frame_index
        )
        for frame_index in representative_frames(
            seed
        )
    ]

    bundle = UnifiedEvidenceBundle(
        bundle_id=(
            f"day21_cross_domain_{seed.event_id}"
        ),
        question=question,
        items=[
            *document_items,
            *robot_items,
        ],
    )
    valid, errors = (
        validate_cross_domain_bundle(
            bundle
        )
    )
    if not valid:
        raise ValueError(
            "invalid Day21 cross-domain bundle: "
            + repr(errors)
        )

    readiness = build_readiness(
        cross_domain_bundle_valid=valid,
        document_item_count=len(
            document_items
        ),
        robot_item_count=len(
            robot_items
        ),
        localization_origin=(
            LOCALIZATION_ORIGIN
        ),
        manual_support_status=(
            MANUAL_SUPPORT_STATUS
        ),
    )

    return (
        bundle,
        robot_items,
        readiness,
    )


def build_artifact(
    *,
    frozen_after_day20_commit: str,
    day20_report_blob_sha1: str,
    day20_report_sha256: str,
    day19_model_sha256: str,
    document_manifest_sha256: str,
    document_visual_manifest_sha256: str,
    manual_query: str,
    manual_retriever_name: str,
    manual_top_k: int,
    manual_candidates: Sequence[Any],
    manual_trace: dict[str, Any] | None,
    case_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    validate_manual_query_is_label_independent(
        manual_query
    )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_status": ARTIFACT_STATUS,
        "scope": (
            "cross_domain_evidence_assembly_and_abstention_"
            "contract_not_physical_root_cause_inference"
        ),
        "provenance": {
            "frozen_after_day20_commit": (
                frozen_after_day20_commit
            ),
            "day20_report_blob_sha1": (
                day20_report_blob_sha1
            ),
            "day20_report_sha256": (
                day20_report_sha256
            ),
            "day19_model_sha256": (
                day19_model_sha256
            ),
            "document_manifest_sha256": (
                document_manifest_sha256
            ),
            "document_visual_manifest_sha256": (
                document_visual_manifest_sha256
            ),
        },
        "manual_retrieval": {
            "query": manual_query,
            "retriever_name": (
                manual_retriever_name
            ),
            "top_k": int(
                manual_top_k
            ),
            "candidates": [
                {
                    "rank": int(
                        candidate.rank
                    ),
                    "raw_score": float(
                        candidate.raw_score
                    ),
                    "page_number": int(
                        candidate.item.payload.page_number
                    ),
                    "evidence_id": (
                        candidate.item.evidence_id
                    ),
                }
                for candidate in manual_candidates
            ],
            "trace": manual_trace,
            "support_status": (
                MANUAL_SUPPORT_STATUS
            ),
            "causal_relevance_ground_truth": None,
        },
        "diagnostic_policy": {
            "physical_root_cause_policy": "abstain",
            "allowed_output": (
                "evidence_summary_and_missing_evidence_only"
            ),
            "forbidden_claims": [
                "target_offset_or_perception",
                "gripper_close_timing",
                "trajectory_execution_deviation",
            ],
        },
        "cases": list(
            case_records
        ),
        "summary": {
            "case_count": len(
                case_records
            ),
            "root_cause_answerable_count": sum(
                bool(
                    case["readiness"][
                        "root_cause_answerable"
                    ]
                )
                for case in case_records
            ),
            "abstain_count": sum(
                case["readiness"][
                    "decision"
                ] == ROOT_CAUSE_DECISION
                for case in case_records
            ),
        },
    }

    assert_no_review_labels_leaked(
        artifact
    )
    return artifact


def validate_case_record_shape(
    case: dict[str, Any],
    *,
    expected_document_items: int,
    expected_robot_items: int,
) -> None:
    if case.get(
        "localization_origin"
    ) != LOCALIZATION_ORIGIN:
        raise ValueError(
            "case localization_origin mismatch"
        )
    if case.get(
        "manual_support_status"
    ) != MANUAL_SUPPORT_STATUS:
        raise ValueError(
            "case manual_support_status mismatch"
        )

    bundle = case.get(
        "evidence_bundle"
    )
    if not isinstance(
        bundle,
        dict,
    ):
        raise ValueError(
            "case evidence_bundle must be an object"
        )

    items = bundle.get(
        "items",
        [],
    )
    document_count = sum(
        item.get("kind") == "document_page"
        for item in items
    )
    robot_count = sum(
        item.get("kind") == "robot_sample"
        for item in items
    )
    if document_count != int(
        expected_document_items
    ):
        raise ValueError(
            "unexpected document evidence count"
        )
    if robot_count != int(
        expected_robot_items
    ):
        raise ValueError(
            "unexpected robot evidence count"
        )

    readiness = case.get(
        "readiness",
        {},
    )
    if readiness.get(
        "root_cause_answerable"
    ) is not False:
        raise ValueError(
            "Day21 must abstain from physical root cause"
        )
    if readiness.get(
        "decision"
    ) != ROOT_CAUSE_DECISION:
        raise ValueError(
            "Day21 readiness decision mismatch"
        )
