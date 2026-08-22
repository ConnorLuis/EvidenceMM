from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CONFIG_SCHEMA = "evidencemm_day18_robot_benchmark_split_config_v1"
SPLIT_SCHEMA = "evidencemm_day18_robot_benchmark_split_v1"
SOURCE_AUDIT_SCHEMA = "evidencemm_day16_source_audit_v2"
HUMAN_GT_SCHEMA = "day16_human_gt_v1"

BENCHMARK_STATUS = (
    "prospective_heldout_split_frozen_after_day17_baseline"
)
SPLIT_SCOPE = "future_day19_plus_model_selection_and_evaluation"

CLEAN_CATEGORY = "clean_reference_candidate"
ANOMALY_CATEGORY = "operation_anomaly"
ELIGIBLE_CATEGORIES = (CLEAN_CATEGORY, ANOMALY_CATEGORY)
ALLOWED_GT_DISPOSITIONS = ("verified", "reviewed_unresolved")

FORBIDDEN_GOLD_BOUNDARY_KEYS = {
    "failure_interval",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "supporting_robot_refs",
    "counterevidence_robot_refs",
}


@dataclass(frozen=True)
class EligibleEpisode:
    episode_id: str
    audit_category: str
    task_success: bool
    operation_anomaly: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAssignment:
    episode_id: str
    audit_category: str
    split: str
    rank_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoldEventMetadata:
    event_id: str
    episode_id: str
    review_disposition: str


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source}:{lineno}: invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"{source}:{lineno}: expected JSON object"
            )
        rows.append(row)
    return rows


def source_audit_category_counts(
    rows: Iterable[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("audit_category", ""))
        counts[category] = counts.get(category, 0) + 1
    return counts


def _require_source_presence(
    row: dict[str, Any],
    *,
    episode_id: str,
) -> None:
    for key in (
        "raw_episode_dir_exists",
        "metadata_exists",
        "samples_csv_exists",
        "front_dir_exists",
        "wrist_dir_exists",
    ):
        if row.get(key) is not True:
            raise ValueError(
                f"{episode_id}: eligible episode requires {key}=true"
            )


def load_eligible_episodes(
    source_audit_path: str | Path,
) -> tuple[list[dict[str, Any]], list[EligibleEpisode]]:
    rows = load_jsonl(source_audit_path)
    seen: set[str] = set()
    eligible: list[EligibleEpisode] = []

    for row in rows:
        if row.get("schema_version") != SOURCE_AUDIT_SCHEMA:
            raise ValueError(
                "unexpected Day16 source-audit schema_version"
            )

        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            raise ValueError("source audit row missing episode_id")
        if episode_id in seen:
            raise ValueError(
                f"duplicate episode_id in source audit: {episode_id}"
            )
        seen.add(episode_id)

        category = str(row.get("audit_category", ""))
        if category not in ELIGIBLE_CATEGORIES:
            continue

        if row.get("technical_valid") is not True:
            raise ValueError(
                f"{episode_id}: eligible episode must be technical_valid"
            )
        if row.get("diagnostic_eligible") is not True:
            raise ValueError(
                f"{episode_id}: eligible episode must be diagnostic_eligible"
            )
        _require_source_presence(
            row,
            episode_id=episode_id,
        )

        operation_anomaly = bool(row.get("operation_anomaly"))
        task_success = bool(row.get("task_success"))

        if category == CLEAN_CATEGORY:
            if operation_anomaly:
                raise ValueError(
                    f"{episode_id}: clean episode cannot be operation_anomaly"
                )
            if not task_success:
                raise ValueError(
                    f"{episode_id}: clean episode must have task_success=true"
                )
            if row.get("demo_quality_valid") is not True:
                raise ValueError(
                    f"{episode_id}: clean episode must be demo_quality_valid"
                )
            if row.get("valid_for_training") is not True:
                raise ValueError(
                    f"{episode_id}: clean episode must be valid_for_training"
                )
        else:
            if not operation_anomaly:
                raise ValueError(
                    f"{episode_id}: anomaly category requires operation_anomaly"
                )

        eligible.append(
            EligibleEpisode(
                episode_id=episode_id,
                audit_category=category,
                task_success=task_success,
                operation_anomaly=operation_anomaly,
            )
        )

    return rows, sorted(
        eligible,
        key=lambda item: item.episode_id,
    )


def split_rank_digest(
    *,
    seed: str,
    audit_category: str,
    episode_id: str,
) -> str:
    if not seed:
        raise ValueError("split seed must be non-empty")
    payload = (
        f"{seed}|{audit_category}|{episode_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_stratified_split(
    episodes: Iterable[EligibleEpisode],
    *,
    seed: str,
    held_out_counts: dict[str, int],
) -> list[SplitAssignment]:
    values = list(episodes)
    if len({item.episode_id for item in values}) != len(values):
        raise ValueError("duplicate episode_id in eligible episodes")

    assignments: list[SplitAssignment] = []

    for category in ELIGIBLE_CATEGORIES:
        members = [
            item
            for item in values
            if item.audit_category == category
        ]
        requested = int(held_out_counts.get(category, -1))
        if requested < 0:
            raise ValueError(
                f"held_out_counts missing {category}"
            )
        if requested >= len(members):
            raise ValueError(
                f"{category}: held-out count must be smaller "
                "than category population"
            )

        ranked = sorted(
            members,
            key=lambda item: (
                split_rank_digest(
                    seed=seed,
                    audit_category=category,
                    episode_id=item.episode_id,
                ),
                item.episode_id,
            ),
        )
        held_out_ids = {
            item.episode_id
            for item in ranked[:requested]
        }

        for item in members:
            assignments.append(
                SplitAssignment(
                    episode_id=item.episode_id,
                    audit_category=category,
                    split=(
                        "held_out"
                        if item.episode_id in held_out_ids
                        else "development"
                    ),
                    rank_digest=split_rank_digest(
                        seed=seed,
                        audit_category=category,
                        episode_id=item.episode_id,
                    ),
                )
            )

    if len(assignments) != len(values):
        unknown = sorted(
            {
                item.audit_category
                for item in values
            }
            - set(ELIGIBLE_CATEGORIES)
        )
        raise ValueError(
            f"unsupported eligible categories: {unknown}"
        )

    return sorted(
        assignments,
        key=lambda item: item.episode_id,
    )


def load_gold_event_metadata(
    human_gt_path: str | Path,
) -> list[GoldEventMetadata]:
    rows = load_jsonl(human_gt_path)
    seen: set[str] = set()
    result: list[GoldEventMetadata] = []

    for row in rows:
        if row.get("schema_version") != HUMAN_GT_SCHEMA:
            raise ValueError(
                "unexpected Day16 human-GT schema_version"
            )
        event_id = str(row.get("event_id", ""))
        episode_id = str(row.get("episode_id", ""))
        disposition = str(
            row.get("review_disposition", "")
        )

        if not event_id or not episode_id:
            raise ValueError(
                "human GT row missing event_id or episode_id"
            )
        if event_id in seen:
            raise ValueError(
                f"duplicate event_id in human GT: {event_id}"
            )
        seen.add(event_id)

        if disposition not in ALLOWED_GT_DISPOSITIONS:
            raise ValueError(
                f"{event_id}: unsupported review_disposition"
            )

        result.append(
            GoldEventMetadata(
                event_id=event_id,
                episode_id=episode_id,
                review_disposition=disposition,
            )
        )

    return sorted(
        result,
        key=lambda item: item.event_id,
    )


def validate_expected_source_counts(
    rows: list[dict[str, Any]],
    expected: dict[str, int],
) -> None:
    if len(rows) != int(expected["total_rows"]):
        raise ValueError(
            "source audit total_rows differs from Day18 config"
        )

    counts = source_audit_category_counts(rows)
    for category in (
        CLEAN_CATEGORY,
        ANOMALY_CATEGORY,
        "demo_quality_only",
        "technical_exclusion",
    ):
        if counts.get(category, 0) != int(
            expected[category]
        ):
            raise ValueError(
                f"source audit count mismatch for {category}: "
                f"expected={expected[category]} "
                f"actual={counts.get(category, 0)}"
            )


def validate_expected_gold_counts(
    events: list[GoldEventMetadata],
    expected: dict[str, int],
) -> None:
    episodes = {event.episode_id for event in events}
    verified = sum(
        event.review_disposition == "verified"
        for event in events
    )
    unresolved = sum(
        event.review_disposition == "reviewed_unresolved"
        for event in events
    )

    actual = {
        "episode_count": len(episodes),
        "event_count": len(events),
        "verified_event_count": verified,
        "reviewed_unresolved_count": unresolved,
    }
    wanted = {
        key: int(expected[key])
        for key in actual
    }
    if actual != wanted:
        raise ValueError(
            f"human GT count mismatch: expected={wanted}, actual={actual}"
        )


def split_counts(
    assignments: Iterable[SplitAssignment],
) -> dict[str, int]:
    values = list(assignments)

    def count(split: str, category: str | None = None) -> int:
        return sum(
            item.split == split
            and (
                category is None
                or item.audit_category == category
            )
            for item in values
        )

    return {
        "eligible_episode_count": len(values),
        "development_episode_count": count("development"),
        "held_out_episode_count": count("held_out"),
        "development_clean_count": count(
            "development",
            CLEAN_CATEGORY,
        ),
        "development_anomaly_count": count(
            "development",
            ANOMALY_CATEGORY,
        ),
        "held_out_clean_count": count(
            "held_out",
            CLEAN_CATEGORY,
        ),
        "held_out_anomaly_count": count(
            "held_out",
            ANOMALY_CATEGORY,
        ),
    }


def validate_expected_split_counts(
    assignments: list[SplitAssignment],
    expected: dict[str, int],
) -> None:
    actual = split_counts(assignments)
    wanted = {
        key: int(expected[key])
        for key in actual
    }
    if actual != wanted:
        raise ValueError(
            f"split count mismatch: expected={wanted}, actual={actual}"
        )


def _event_summary_by_episode(
    events: Iterable[GoldEventMetadata],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[GoldEventMetadata]] = {}
    for event in events:
        grouped.setdefault(
            event.episode_id,
            [],
        ).append(event)

    result: dict[str, dict[str, Any]] = {}
    for episode_id, episode_events in grouped.items():
        result[episode_id] = {
            "human_gt_event_count": len(episode_events),
            "human_gt_event_ids": sorted(
                event.event_id
                for event in episode_events
            ),
            "human_gt_review_dispositions": sorted(
                {
                    event.review_disposition
                    for event in episode_events
                }
            ),
        }
    return result


def _split_gold_counts(
    assignments: list[SplitAssignment],
    events: list[GoldEventMetadata],
) -> dict[str, dict[str, int]]:
    split_by_episode = {
        item.episode_id: item.split
        for item in assignments
    }
    result = {
        "development": {
            "event_count": 0,
            "verified_event_count": 0,
            "reviewed_unresolved_count": 0,
        },
        "held_out": {
            "event_count": 0,
            "verified_event_count": 0,
            "reviewed_unresolved_count": 0,
        },
    }

    for event in events:
        split = split_by_episode.get(event.episode_id)
        if split is None:
            raise ValueError(
                f"{event.event_id}: human-GT episode is not split-eligible"
            )
        bucket = result[split]
        bucket["event_count"] += 1
        if event.review_disposition == "verified":
            bucket["verified_event_count"] += 1
        else:
            bucket["reviewed_unresolved_count"] += 1

    return result


def _split_episode_ids(
    assignments: Iterable[SplitAssignment],
    split: str,
) -> list[str]:
    return sorted(
        item.episode_id
        for item in assignments
        if item.split == split
    )


def _assert_no_forbidden_gold_boundaries(
    value: Any,
    *,
    path: str = "$",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_GOLD_BOUNDARY_KEYS:
                raise ValueError(
                    f"split artifact leaks gold boundary field "
                    f"{path}.{key}"
                )
            _assert_no_forbidden_gold_boundaries(
                child,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_gold_boundaries(
                child,
                path=f"{path}[{index}]",
            )


def build_split_artifact(
    *,
    eligible_episodes: list[EligibleEpisode],
    assignments: list[SplitAssignment],
    gold_events: list[GoldEventMetadata],
    source_audit_path: str,
    source_audit_sha256: str,
    human_gt_path: str,
    human_gt_sha256: str,
    seed: str,
    held_out_counts: dict[str, int],
    frozen_after_day17_commit: str,
) -> dict[str, Any]:
    assignment_by_id = {
        item.episode_id: item
        for item in assignments
    }
    eligible_by_id = {
        item.episode_id: item
        for item in eligible_episodes
    }

    if set(assignment_by_id) != set(eligible_by_id):
        raise ValueError(
            "split assignments do not match eligible episode universe"
        )

    gold_episode_ids = {
        event.episode_id
        for event in gold_events
    }
    anomaly_ids = {
        item.episode_id
        for item in eligible_episodes
        if item.audit_category == ANOMALY_CATEGORY
    }
    if gold_episode_ids != anomaly_ids:
        raise ValueError(
            "human-GT episode universe must match anomaly episode universe"
        )

    event_summary = _event_summary_by_episode(gold_events)

    episode_rows = []
    for episode_id in sorted(eligible_by_id):
        episode = eligible_by_id[episode_id]
        assignment = assignment_by_id[episode_id]
        summary = event_summary.get(
            episode_id,
            {
                "human_gt_event_count": 0,
                "human_gt_event_ids": [],
                "human_gt_review_dispositions": [],
            },
        )
        episode_rows.append(
            {
                "episode_id": episode_id,
                "split": assignment.split,
                "source_category": episode.audit_category,
                "task_success": episode.task_success,
                "operation_anomaly": episode.operation_anomaly,
                **summary,
            }
        )

    artifact = {
        "schema_version": SPLIT_SCHEMA,
        "benchmark_status": BENCHMARK_STATUS,
        "split_scope": SPLIT_SCOPE,
        "provenance": {
            "frozen_after_day17_commit": (
                frozen_after_day17_commit
            ),
            "source_audit_path": source_audit_path,
            "source_audit_sha256": source_audit_sha256,
            "human_gt_path": human_gt_path,
            "human_gt_sha256": human_gt_sha256,
        },
        "anti_leakage": {
            "membership_inputs": [
                "episode_id",
                "audit_category",
            ],
            "human_gt_used_for_membership": False,
            "failure_interval_used_for_membership": False,
            "membership_frozen_before_human_gt_metadata_load": True,
            "split_artifact_contains_failure_boundaries": False,
            "note": (
                "Day17 already evaluated all reviewed failures; "
                "this split is prospective only for Day19+ model "
                "selection/evaluation and does not retroactively make "
                "Day17 held-out."
            ),
        },
        "protocol": {
            "seed": seed,
            "rank_rule": (
                "sha256(seed|audit_category|episode_id)"
            ),
            "stratification": list(ELIGIBLE_CATEGORIES),
            "held_out_counts": {
                category: int(held_out_counts[category])
                for category in ELIGIBLE_CATEGORIES
            },
        },
        "counts": split_counts(assignments),
        "gold_metadata_counts_by_split": _split_gold_counts(
            assignments,
            gold_events,
        ),
        "splits": {
            "development": {
                "episode_ids": _split_episode_ids(
                    assignments,
                    "development",
                ),
            },
            "held_out": {
                "episode_ids": _split_episode_ids(
                    assignments,
                    "held_out",
                ),
            },
        },
        "episodes": episode_rows,
    }

    _assert_no_forbidden_gold_boundaries(artifact)
    return artifact


def validate_split_artifact_no_gold_boundaries(
    artifact: dict[str, Any],
) -> None:
    _assert_no_forbidden_gold_boundaries(artifact)
    anti = artifact.get("anti_leakage", {})
    if anti.get("split_artifact_contains_failure_boundaries") is not False:
        raise ValueError(
            "split artifact must declare absence of failure boundaries"
        )


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
