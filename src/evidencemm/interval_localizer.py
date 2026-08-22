from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CONFIG_SCHEMA = "evidencemm_day19_interval_localizer_config_v1"
MODEL_SCHEMA = "evidencemm_day19_interval_localizer_model_v1"
REPORT_SCHEMA = "evidencemm_day19_interval_localizer_report_v1"

MODEL_STATUS = "development_selected_interval_proposal_localizer"
MODEL_SELECTION_SPLIT = "development"
CENTER_POLICY = "frozen_day16_signal_candidates_excluding_uniform_only"
INTERVAL_RULE = "symmetric_radius_around_candidate_center"

UNIFORM_REASON = "uniform_anchor"
VERIFIED = "verified"
REVIEWED_UNRESOLVED = "reviewed_unresolved"

FORBIDDEN_MODEL_KEYS = {
    "failure_interval",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "supporting_robot_refs",
    "counterevidence_robot_refs",
}


@dataclass(frozen=True)
class DevelopmentGoldEvent:
    event_id: str
    episode_id: str
    observed_failure_mode: str
    review_disposition: str
    start_frame: int | None
    end_frame: int | None

    @property
    def is_verified(self) -> bool:
        return self.review_disposition == VERIFIED


@dataclass(frozen=True)
class IntervalProposal:
    center_frame: int
    start_frame: int
    end_frame: int
    radius_frames: int
    center_timestamp_sec: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class EventIntervalResult:
    event_id: str
    episode_id: str
    observed_failure_mode: str
    gold_start_frame: int
    gold_end_frame: int
    best_proposal_center_frame: int
    best_proposal_start_frame: int
    best_proposal_end_frame: int
    best_iou: float
    overlap_hit: bool
    onset_abs_error_frames: int
    offset_abs_error_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def load_day18_split(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if artifact.get("schema_version") != (
        "evidencemm_day18_robot_benchmark_split_v1"
    ):
        raise ValueError(
            "unexpected Day18 benchmark split schema_version"
        )
    return artifact


def split_episode_sets(
    split_artifact: dict[str, Any],
) -> tuple[set[str], set[str], set[str], set[str]]:
    development = set(
        split_artifact["splits"]["development"]["episode_ids"]
    )
    held_out = set(
        split_artifact["splits"]["held_out"]["episode_ids"]
    )
    if development & held_out:
        raise ValueError(
            "development and held-out episode sets overlap"
        )

    episode_rows = split_artifact["episodes"]
    row_ids = {
        str(row["episode_id"])
        for row in episode_rows
    }
    if row_ids != development | held_out:
        raise ValueError(
            "Day18 episode rows differ from split episode universe"
        )

    development_anomaly = {
        str(row["episode_id"])
        for row in episode_rows
        if (
            row["split"] == "development"
            and row["source_category"] == "operation_anomaly"
        )
    }
    held_out_anomaly = {
        str(row["episode_id"])
        for row in episode_rows
        if (
            row["split"] == "held_out"
            and row["source_category"] == "operation_anomaly"
        )
    }

    return (
        development,
        held_out,
        development_anomaly,
        held_out_anomaly,
    )


def validate_expected_split_counts(
    split_artifact: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    (
        development,
        held_out,
        development_anomaly,
        held_out_anomaly,
    ) = split_episode_sets(split_artifact)

    actual = {
        "development_episode_count": len(development),
        "held_out_episode_count": len(held_out),
        "development_anomaly_episode_count": len(
            development_anomaly
        ),
        "held_out_anomaly_episode_count": len(
            held_out_anomaly
        ),
    }
    wanted = {
        key: int(expected[key])
        for key in actual
    }
    if actual != wanted:
        raise ValueError(
            f"Day18 split count mismatch: expected={wanted}, "
            f"actual={actual}"
        )


def load_development_gold_events(
    path: str | Path,
    *,
    allowed_episode_ids: set[str],
) -> list[DevelopmentGoldEvent]:
    source = Path(path)
    events: list[DevelopmentGoldEvent] = []
    seen: set[str] = set()

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

        episode_id = str(row.get("episode_id", ""))
        if episode_id not in allowed_episode_ids:
            # Prospective anti-leakage rule: held-out interval fields
            # are never accessed by Day19 model-selection logic.
            continue

        event_id = str(row.get("event_id", ""))
        if not event_id:
            raise ValueError(
                f"{source}:{lineno}: missing event_id"
            )
        if event_id in seen:
            raise ValueError(
                f"duplicate development event_id={event_id}"
            )
        seen.add(event_id)

        disposition = str(
            row.get("review_disposition", "")
        )
        if disposition == VERIFIED:
            interval = row.get("failure_interval")
            if not isinstance(interval, dict):
                raise ValueError(
                    f"{event_id}: verified event requires interval"
                )
            start_frame = int(interval["start_frame"])
            end_frame = int(interval["end_frame"])
            if start_frame > end_frame:
                raise ValueError(
                    f"{event_id}: invalid failure interval"
                )
        elif disposition == REVIEWED_UNRESOLVED:
            if row.get("failure_interval") is not None:
                raise ValueError(
                    f"{event_id}: unresolved event must not have interval"
                )
            start_frame = None
            end_frame = None
        else:
            raise ValueError(
                f"{event_id}: unsupported review_disposition"
            )

        events.append(
            DevelopmentGoldEvent(
                event_id=event_id,
                episode_id=episode_id,
                observed_failure_mode=str(
                    row["observed_failure_mode"]
                ),
                review_disposition=disposition,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    return sorted(
        events,
        key=lambda item: item.event_id,
    )


def validate_expected_development_gold(
    events: Sequence[DevelopmentGoldEvent],
    expected: dict[str, Any],
) -> None:
    verified = [
        event
        for event in events
        if event.is_verified
    ]
    unresolved = [
        event
        for event in events
        if event.review_disposition
        == REVIEWED_UNRESOLVED
    ]
    actual = {
        "development_event_count": len(events),
        "development_verified_event_count": len(verified),
        "development_reviewed_unresolved_count": len(
            unresolved
        ),
    }
    wanted = {
        key: int(expected[key])
        for key in actual
    }
    if actual != wanted:
        raise ValueError(
            f"development human-GT mismatch: "
            f"expected={wanted}, actual={actual}"
        )


def signal_candidates(
    candidates: Sequence[Any],
) -> list[Any]:
    result = []
    for candidate in candidates:
        reasons = tuple(
            str(reason)
            for reason in candidate.reasons
        )
        if not reasons:
            raise ValueError(
                "candidate must contain at least one reason"
            )
        if all(
            reason == UNIFORM_REASON
            for reason in reasons
        ):
            continue
        result.append(candidate)

    if not result:
        raise ValueError(
            "signal-candidate filtering removed all candidates"
        )

    frames = [
        int(candidate.frame_index)
        for candidate in result
    ]
    if len(frames) != len(set(frames)):
        raise ValueError(
            "duplicate signal candidate frame"
        )
    return sorted(
        result,
        key=lambda candidate: int(candidate.frame_index),
    )


def build_interval_proposals(
    candidates: Sequence[Any],
    *,
    frame_count: int,
    radius_frames: int,
) -> list[IntervalProposal]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    radius = int(radius_frames)
    if radius < 0:
        raise ValueError(
            "radius_frames must be non-negative"
        )

    centers = signal_candidates(candidates)
    proposals = []
    for candidate in centers:
        center = int(candidate.frame_index)
        if not 0 <= center < frame_count:
            raise ValueError(
                f"candidate center outside episode: {center}"
            )
        proposals.append(
            IntervalProposal(
                center_frame=center,
                start_frame=max(0, center - radius),
                end_frame=min(
                    frame_count - 1,
                    center + radius,
                ),
                radius_frames=radius,
                center_timestamp_sec=float(
                    candidate.timestamp_sec
                ),
                reasons=tuple(
                    sorted(
                        str(reason)
                        for reason in candidate.reasons
                        if reason != UNIFORM_REASON
                    )
                ),
            )
        )
    return proposals


def interval_iou(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:
    if start_a > end_a or start_b > end_b:
        raise ValueError(
            "interval start must be <= interval end"
        )

    intersection = max(
        0,
        min(end_a, end_b)
        - max(start_a, start_b)
        + 1,
    )
    if intersection == 0:
        return 0.0

    size_a = end_a - start_a + 1
    size_b = end_b - start_b + 1
    union = size_a + size_b - intersection
    return intersection / union


def evaluate_verified_event(
    *,
    event: DevelopmentGoldEvent,
    proposals: Sequence[IntervalProposal],
) -> EventIntervalResult:
    if not event.is_verified:
        raise ValueError(
            f"{event.event_id}: only verified event may be scored"
        )
    if (
        event.start_frame is None
        or event.end_frame is None
    ):
        raise ValueError(
            f"{event.event_id}: verified interval missing"
        )
    if not proposals:
        raise ValueError(
            f"{event.episode_id}: proposal list must be non-empty"
        )

    ranked = sorted(
        (
            (
                interval_iou(
                    proposal.start_frame,
                    proposal.end_frame,
                    event.start_frame,
                    event.end_frame,
                ),
                abs(
                    proposal.center_frame
                    - (
                        event.start_frame
                        + event.end_frame
                    )
                    / 2.0
                ),
                proposal.center_frame,
                proposal,
            )
            for proposal in proposals
        ),
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
        ),
    )
    best_iou, _, _, best = ranked[0]

    return EventIntervalResult(
        event_id=event.event_id,
        episode_id=event.episode_id,
        observed_failure_mode=event.observed_failure_mode,
        gold_start_frame=event.start_frame,
        gold_end_frame=event.end_frame,
        best_proposal_center_frame=best.center_frame,
        best_proposal_start_frame=best.start_frame,
        best_proposal_end_frame=best.end_frame,
        best_iou=best_iou,
        overlap_hit=(best_iou > 0.0),
        onset_abs_error_frames=abs(
            best.start_frame - event.start_frame
        ),
        offset_abs_error_frames=abs(
            best.end_frame - event.end_frame
        ),
    )


def _metric_block(
    results: Sequence[EventIntervalResult],
    *,
    iou_thresholds: Sequence[float],
) -> dict[str, Any]:
    if not results:
        return {
            "event_count": 0,
            "event_recall": None,
            "mean_best_iou": None,
            "median_best_iou": None,
            "recall_at_iou": {
                _threshold_key(value): None
                for value in iou_thresholds
            },
            "mean_onset_abs_error_frames": None,
            "mean_offset_abs_error_frames": None,
        }

    count = len(results)
    return {
        "event_count": count,
        "event_recall": (
            sum(result.overlap_hit for result in results)
            / count
        ),
        "mean_best_iou": statistics.fmean(
            result.best_iou
            for result in results
        ),
        "median_best_iou": statistics.median(
            result.best_iou
            for result in results
        ),
        "recall_at_iou": {
            _threshold_key(threshold): (
                sum(
                    result.best_iou >= threshold
                    for result in results
                )
                / count
            )
            for threshold in iou_thresholds
        },
        "mean_onset_abs_error_frames": (
            statistics.fmean(
                result.onset_abs_error_frames
                for result in results
            )
        ),
        "mean_offset_abs_error_frames": (
            statistics.fmean(
                result.offset_abs_error_frames
                for result in results
            )
        ),
    }


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def evaluate_radius(
    *,
    radius_frames: int,
    candidate_sets: dict[str, Sequence[Any]],
    frame_counts: dict[str, int],
    verified_events: Sequence[DevelopmentGoldEvent],
    iou_thresholds: Sequence[float],
) -> tuple[dict[str, Any], list[EventIntervalResult]]:
    proposals_by_episode: dict[
        str,
        list[IntervalProposal],
    ] = {}

    for episode_id, candidates in candidate_sets.items():
        if episode_id not in frame_counts:
            raise ValueError(
                f"{episode_id}: missing frame_count"
            )
        proposals_by_episode[episode_id] = (
            build_interval_proposals(
                candidates,
                frame_count=int(
                    frame_counts[episode_id]
                ),
                radius_frames=radius_frames,
            )
        )

    results = []
    for event in verified_events:
        proposals = proposals_by_episode.get(
            event.episode_id
        )
        if proposals is None:
            raise ValueError(
                f"{event.event_id}: no development proposal set"
            )
        results.append(
            evaluate_verified_event(
                event=event,
                proposals=proposals,
            )
        )

    metrics = _metric_block(
        results,
        iou_thresholds=iou_thresholds,
    )
    proposal_counts = [
        len(values)
        for values in proposals_by_episode.values()
    ]
    metrics["radius_frames"] = int(radius_frames)
    metrics["episode_count"] = len(
        proposals_by_episode
    )
    metrics["mean_proposals_per_episode"] = (
        statistics.fmean(proposal_counts)
        if proposal_counts
        else 0.0
    )
    metrics["min_proposals_per_episode"] = (
        min(proposal_counts)
        if proposal_counts
        else 0
    )
    metrics["max_proposals_per_episode"] = (
        max(proposal_counts)
        if proposal_counts
        else 0
    )

    return metrics, results


def selection_key(
    metrics: dict[str, Any],
) -> tuple[float, float, float, int]:
    recall_at_025 = float(
        metrics["recall_at_iou"][
            _threshold_key(0.25)
        ]
    )
    return (
        float(metrics["event_recall"]),
        recall_at_025,
        float(metrics["mean_best_iou"]),
        -int(metrics["radius_frames"]),
    )


def select_radius(
    *,
    radius_grid_frames: Iterable[int],
    candidate_sets: dict[str, Sequence[Any]],
    frame_counts: dict[str, int],
    verified_events: Sequence[DevelopmentGoldEvent],
    iou_thresholds: Sequence[float],
) -> tuple[
    int,
    list[dict[str, Any]],
    list[EventIntervalResult],
]:
    radii = sorted(
        {
            int(value)
            for value in radius_grid_frames
        }
    )
    if not radii:
        raise ValueError(
            "radius_grid_frames must be non-empty"
        )
    if any(value < 0 for value in radii):
        raise ValueError(
            "radius_grid_frames must be non-negative"
        )

    thresholds = sorted(
        {
            float(value)
            for value in iou_thresholds
        }
    )
    if 0.25 not in thresholds:
        raise ValueError(
            "selection objective requires IoU threshold 0.25"
        )
    if any(
        value <= 0.0 or value > 1.0
        for value in thresholds
    ):
        raise ValueError(
            "IoU thresholds must lie in (0, 1]"
        )

    grid_metrics: list[dict[str, Any]] = []
    results_by_radius: dict[
        int,
        list[EventIntervalResult],
    ] = {}

    for radius in radii:
        metrics, results = evaluate_radius(
            radius_frames=radius,
            candidate_sets=candidate_sets,
            frame_counts=frame_counts,
            verified_events=verified_events,
            iou_thresholds=thresholds,
        )
        grid_metrics.append(metrics)
        results_by_radius[radius] = results

    selected = max(
        grid_metrics,
        key=selection_key,
    )
    selected_radius = int(
        selected["radius_frames"]
    )
    return (
        selected_radius,
        grid_metrics,
        results_by_radius[selected_radius],
    )


def build_model_artifact(
    *,
    split_sha256: str,
    selector_config_sha256: str,
    human_gt_sha256: str,
    frozen_after_day18_commit: str,
    development_episode_count: int,
    development_anomaly_episode_count: int,
    held_out_episode_count: int,
    held_out_anomaly_episode_count: int,
    development_verified_event_count: int,
    development_reviewed_unresolved_count: int,
    radius_grid_frames: Sequence[int],
    iou_thresholds: Sequence[float],
    selected_radius_frames: int,
    grid_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected_metrics = next(
        metrics
        for metrics in grid_metrics
        if int(metrics["radius_frames"])
        == int(selected_radius_frames)
    )

    artifact = {
        "schema_version": MODEL_SCHEMA,
        "model_status": MODEL_STATUS,
        "model_selection_split": MODEL_SELECTION_SPLIT,
        "provenance": {
            "frozen_after_day18_commit": (
                frozen_after_day18_commit
            ),
            "benchmark_split_sha256": split_sha256,
            "frozen_selector_config_sha256": (
                selector_config_sha256
            ),
            "human_gt_sha256": human_gt_sha256,
        },
        "anti_leakage": {
            "candidate_generation_uses_gold": False,
            "held_out_gt_used_for_model_selection": False,
            "held_out_metrics_reported": False,
            "held_out_episode_ids_materialized_in_model_artifact": False,
            "note": (
                "Day18 is prospective after all Day16 reviews existed; "
                "Day19 enforces code-level development-only model selection "
                "and defers held-out scoring to a later stage."
            ),
        },
        "localizer": {
            "center_policy": CENTER_POLICY,
            "interval_rule": INTERVAL_RULE,
            "selected_radius_frames": int(
                selected_radius_frames
            ),
        },
        "development_selection": {
            "development_episode_count": int(
                development_episode_count
            ),
            "development_anomaly_episode_count": int(
                development_anomaly_episode_count
            ),
            "held_out_episode_count_not_evaluated": int(
                held_out_episode_count
            ),
            "held_out_anomaly_episode_count_not_evaluated": int(
                held_out_anomaly_episode_count
            ),
            "verified_event_count": int(
                development_verified_event_count
            ),
            "reviewed_unresolved_count": int(
                development_reviewed_unresolved_count
            ),
            "radius_grid_frames": [
                int(value)
                for value in radius_grid_frames
            ],
            "iou_thresholds": [
                float(value)
                for value in iou_thresholds
            ],
            "selection_objective": [
                "maximize_event_recall",
                "maximize_recall_at_iou_0p25",
                "maximize_mean_best_iou",
                "minimize_radius_frames",
            ],
            "selected_metrics": dict(
                selected_metrics
            ),
            "grid_metrics": [
                dict(metrics)
                for metrics in grid_metrics
            ],
        },
    }
    validate_model_artifact_no_gold_boundaries(
        artifact
    )
    return artifact


def build_development_report(
    *,
    selected_radius_frames: int,
    grid_metrics: Sequence[dict[str, Any]],
    selected_results: Sequence[EventIntervalResult],
    unresolved_events: Sequence[DevelopmentGoldEvent],
    candidate_sets: dict[str, Sequence[Any]],
    frame_counts: dict[str, int],
    iou_thresholds: Sequence[float],
) -> dict[str, Any]:
    proposals = {
        episode_id: [
            proposal.to_dict()
            for proposal in build_interval_proposals(
                candidates,
                frame_count=frame_counts[episode_id],
                radius_frames=selected_radius_frames,
            )
        ]
        for episode_id, candidates in sorted(
            candidate_sets.items()
        )
    }

    return {
        "schema_version": REPORT_SCHEMA,
        "evaluation_scope": "development_only",
        "held_out_evaluated": False,
        "center_policy": CENTER_POLICY,
        "interval_rule": INTERVAL_RULE,
        "selected_radius_frames": int(
            selected_radius_frames
        ),
        "iou_thresholds": [
            float(value)
            for value in iou_thresholds
        ],
        "grid_metrics": [
            dict(metrics)
            for metrics in grid_metrics
        ],
        "development_proposals": proposals,
        "event_results": [
            result.to_dict()
            for result in selected_results
        ],
        "reviewed_unresolved_event_ids": sorted(
            event.event_id
            for event in unresolved_events
        ),
    }


def _assert_no_forbidden_model_keys(
    value: Any,
    *,
    path: str = "$",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_MODEL_KEYS:
                raise ValueError(
                    f"model artifact leaks development/held-out "
                    f"boundary field {path}.{key}"
                )
            _assert_no_forbidden_model_keys(
                child,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_model_keys(
                child,
                path=f"{path}[{index}]",
            )


def validate_model_artifact_no_gold_boundaries(
    artifact: dict[str, Any],
) -> None:
    _assert_no_forbidden_model_keys(artifact)
    anti = artifact.get("anti_leakage", {})
    if anti.get(
        "held_out_gt_used_for_model_selection"
    ) is not False:
        raise ValueError(
            "model artifact must declare held-out GT unused"
        )
    if anti.get("held_out_metrics_reported") is not False:
        raise ValueError(
            "Day19 must not report held-out metrics"
        )
