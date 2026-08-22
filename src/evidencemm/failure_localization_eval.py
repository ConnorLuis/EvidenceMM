from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from evidencemm.data_binding import sha256_file
from evidencemm.review_pack import (
    SelectionConfig,
    build_state_scores,
    select_review_candidates,
    visual_motion_scores,
)
from evidencemm.state_action_selection import (
    load_state_action_samples,
    validate_source_semantics,
)
from evidencemm.temporal_evidence import (
    EpisodeManifest,
    FrameRecord,
    load_frame_records,
)


REPORT_SCHEMA = "evidencemm_day17_failure_localization_report_v1"
BENCHMARK_STATUS = (
    "small_real_failure_localization_diagnostic_not_held_out"
)
SELECTOR_STATUS = "frozen_day16_pre_gt_review_selector"


@dataclass(frozen=True)
class GoldFailureEvent:
    event_id: str
    episode_id: str
    observed_failure_mode: str
    review_disposition: str
    start_frame: int | None
    end_frame: int | None
    start_sec: float | None
    end_sec: float | None

    @property
    def is_verified(self) -> bool:
        return self.review_disposition == "verified"


@dataclass(frozen=True)
class CandidateFrame:
    frame_index: int
    timestamp_sec: float
    reasons: tuple[str, ...]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class EventLocalizationResult:
    event_id: str
    episode_id: str
    observed_failure_mode: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    exact_hit: bool
    tolerance_hits: dict[str, bool]
    min_frame_distance: int
    min_time_distance_ms: float
    closest_candidate_frame: int
    closest_candidate_timestamp_sec: float
    candidate_frames_inside_interval: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_frames_inside_interval"] = list(
            self.candidate_frames_inside_interval
        )
        return payload


def load_human_gt_events(path: str | Path) -> list[GoldFailureEvent]:
    source = Path(path)
    events: list[GoldFailureEvent] = []

    for lineno, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source}:{lineno}: invalid JSON"
            ) from exc

        interval = row.get("failure_interval")
        disposition = str(row.get("review_disposition", ""))

        if disposition == "verified":
            if not isinstance(interval, dict):
                raise ValueError(
                    f"{row.get('event_id')}: verified event "
                    "requires failure_interval"
                )
            start_frame = int(interval["start_frame"])
            end_frame = int(interval["end_frame"])
            start_sec = float(interval["start_sec"])
            end_sec = float(interval["end_sec"])
        elif disposition == "reviewed_unresolved":
            if interval is not None:
                raise ValueError(
                    f"{row.get('event_id')}: unresolved event "
                    "must not contain failure_interval"
                )
            start_frame = None
            end_frame = None
            start_sec = None
            end_sec = None
        else:
            raise ValueError(
                f"{row.get('event_id')}: unsupported "
                f"review_disposition={disposition!r}"
            )

        events.append(
            GoldFailureEvent(
                event_id=str(row["event_id"]),
                episode_id=str(row["episode_id"]),
                observed_failure_mode=str(
                    row["observed_failure_mode"]
                ),
                review_disposition=disposition,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    return events


def validate_gold_universe(
    events: Iterable[GoldFailureEvent],
    *,
    expected_episode_count: int,
    expected_event_count: int,
    expected_verified_count: int,
    expected_unresolved_count: int,
) -> None:
    values = list(events)
    episode_ids = {event.episode_id for event in values}
    event_ids = [event.event_id for event in values]

    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate event_id in human GT")

    verified = [event for event in values if event.is_verified]
    unresolved = [
        event
        for event in values
        if event.review_disposition == "reviewed_unresolved"
    ]

    expected = {
        "episode_count": expected_episode_count,
        "event_count": expected_event_count,
        "verified_event_count": expected_verified_count,
        "reviewed_unresolved_count": expected_unresolved_count,
    }
    actual = {
        "episode_count": len(episode_ids),
        "event_count": len(values),
        "verified_event_count": len(verified),
        "reviewed_unresolved_count": len(unresolved),
    }

    if actual != expected:
        raise ValueError(
            f"human GT universe mismatch: expected={expected}, "
            f"actual={actual}"
        )


def interval_distance_frames(
    frame_index: int,
    start_frame: int,
    end_frame: int,
) -> int:
    if start_frame > end_frame:
        raise ValueError("start_frame must be <= end_frame")
    if frame_index < start_frame:
        return start_frame - frame_index
    if frame_index > end_frame:
        return frame_index - end_frame
    return 0


def interval_distance_seconds(
    timestamp_sec: float,
    start_sec: float,
    end_sec: float,
) -> float:
    if start_sec > end_sec:
        raise ValueError("start_sec must be <= end_sec")
    if timestamp_sec < start_sec:
        return start_sec - timestamp_sec
    if timestamp_sec > end_sec:
        return timestamp_sec - end_sec
    return 0.0


def evaluate_verified_event(
    *,
    event: GoldFailureEvent,
    candidates: list[CandidateFrame],
    tolerance_frames: Iterable[int],
) -> EventLocalizationResult:
    if not event.is_verified:
        raise ValueError(
            f"{event.event_id}: only verified events are scored"
        )
    if not candidates:
        raise ValueError(
            f"{event.episode_id}: candidate list must be non-empty"
        )
    if (
        event.start_frame is None
        or event.end_frame is None
        or event.start_sec is None
        or event.end_sec is None
    ):
        raise ValueError(
            f"{event.event_id}: verified event interval is incomplete"
        )

    unique_frames = {candidate.frame_index for candidate in candidates}
    if len(unique_frames) != len(candidates):
        raise ValueError(
            f"{event.episode_id}: duplicate candidate frame"
        )

    closest = min(
        candidates,
        key=lambda candidate: (
            interval_distance_frames(
                candidate.frame_index,
                event.start_frame,
                event.end_frame,
            ),
            candidate.frame_index,
        ),
    )

    min_frame_distance = interval_distance_frames(
        closest.frame_index,
        event.start_frame,
        event.end_frame,
    )
    min_time_distance_ms = (
        interval_distance_seconds(
            closest.timestamp_sec,
            event.start_sec,
            event.end_sec,
        )
        * 1000.0
    )

    inside = tuple(
        candidate.frame_index
        for candidate in candidates
        if (
            event.start_frame
            <= candidate.frame_index
            <= event.end_frame
        )
    )

    tolerance_hits: dict[str, bool] = {}
    for tolerance in tolerance_frames:
        value = int(tolerance)
        if value < 0:
            raise ValueError(
                "tolerance_frames must be non-negative"
            )
        tolerance_hits[str(value)] = (
            min_frame_distance <= value
        )

    return EventLocalizationResult(
        event_id=event.event_id,
        episode_id=event.episode_id,
        observed_failure_mode=event.observed_failure_mode,
        start_frame=event.start_frame,
        end_frame=event.end_frame,
        start_sec=event.start_sec,
        end_sec=event.end_sec,
        exact_hit=(min_frame_distance == 0),
        tolerance_hits=tolerance_hits,
        min_frame_distance=min_frame_distance,
        min_time_distance_ms=min_time_distance_ms,
        closest_candidate_frame=closest.frame_index,
        closest_candidate_timestamp_sec=closest.timestamp_sec,
        candidate_frames_inside_interval=inside,
    )


def _metric_block(
    results: list[EventLocalizationResult],
    *,
    tolerances: list[int],
) -> dict[str, Any]:
    if not results:
        return {
            "event_count": 0,
            "exact_event_recall": None,
            "tolerance_event_recall": {
                str(value): None
                for value in tolerances
            },
            "mean_min_frame_distance": None,
            "median_min_frame_distance": None,
            "mean_min_time_distance_ms": None,
            "median_min_time_distance_ms": None,
        }

    count = len(results)
    exact_hits = sum(result.exact_hit for result in results)

    return {
        "event_count": count,
        "exact_event_recall": exact_hits / count,
        "tolerance_event_recall": {
            str(value): (
                sum(
                    result.tolerance_hits[str(value)]
                    for result in results
                )
                / count
            )
            for value in tolerances
        },
        "mean_min_frame_distance": statistics.fmean(
            result.min_frame_distance
            for result in results
        ),
        "median_min_frame_distance": statistics.median(
            result.min_frame_distance
            for result in results
        ),
        "mean_min_time_distance_ms": statistics.fmean(
            result.min_time_distance_ms
            for result in results
        ),
        "median_min_time_distance_ms": statistics.median(
            result.min_time_distance_ms
            for result in results
        ),
    }


def aggregate_localization_results(
    *,
    results: list[EventLocalizationResult],
    unresolved_events: list[GoldFailureEvent],
    tolerance_frames: Iterable[int],
) -> dict[str, Any]:
    tolerances = sorted({int(value) for value in tolerance_frames})
    if any(value < 0 for value in tolerances):
        raise ValueError(
            "tolerance_frames must be non-negative"
        )

    by_mode: dict[str, list[EventLocalizationResult]] = {}
    for result in results:
        by_mode.setdefault(
            result.observed_failure_mode,
            [],
        ).append(result)

    return {
        "verified_event_count": len(results),
        "reviewed_unresolved_count": len(unresolved_events),
        "supervised_interval_event_count": len(results),
        "overall": _metric_block(
            results,
            tolerances=tolerances,
        ),
        "by_observed_failure_mode": {
            mode: _metric_block(
                mode_results,
                tolerances=tolerances,
            )
            for mode, mode_results in sorted(by_mode.items())
        },
        "reviewed_unresolved_event_ids": sorted(
            event.event_id
            for event in unresolved_events
        ),
    }


def _records_by_camera(
    records: list[FrameRecord],
    *,
    episode_id: str,
    frame_count: int,
) -> dict[str, dict[int, FrameRecord]]:
    result: dict[str, dict[int, FrameRecord]] = {
        "front": {},
        "wrist": {},
    }

    for record in records:
        if record.episode_id != episode_id:
            raise ValueError(
                "frame record episode_id differs from manifest"
            )
        if record.camera not in result:
            raise ValueError(
                f"unsupported camera={record.camera!r}"
            )
        if record.frame_index in result[record.camera]:
            raise ValueError(
                f"duplicate {record.camera} frame "
                f"{record.frame_index}"
            )
        result[record.camera][record.frame_index] = record

    expected = set(range(frame_count))
    for camera in ("front", "wrist"):
        if set(result[camera]) != expected:
            raise ValueError(
                f"{episode_id}: {camera} records do not "
                "cover episode exactly"
            )

    return result


def build_frozen_review_candidates(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    episode_id: str,
    diagnostic_manifest_path: str | Path,
    frame_records_path: str | Path,
    selection_config: SelectionConfig,
) -> list[CandidateFrame]:
    project = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    episode_dir = dataset / episode_id

    manifest_path = Path(diagnostic_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = project / manifest_path
    manifest = EpisodeManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.episode_id != episode_id:
        raise ValueError(
            "diagnostic manifest episode_id mismatch"
        )

    metadata_path = episode_dir / "metadata.json"
    samples_path = episode_dir / "samples.csv"

    if sha256_file(metadata_path) != manifest.metadata_sha256:
        raise ValueError(
            f"{episode_id}: metadata.json SHA256 mismatch"
        )
    if sha256_file(samples_path) != manifest.samples_csv_sha256:
        raise ValueError(
            f"{episode_id}: samples.csv SHA256 mismatch"
        )

    validate_source_semantics(metadata_path)
    samples = load_state_action_samples(
        samples_path,
        verify_tracking_error=True,
    )
    if len(samples) != manifest.frame_count:
        raise ValueError(
            f"{episode_id}: sample count differs from manifest"
        )

    records_path = Path(frame_records_path)
    if not records_path.is_absolute():
        records_path = project / records_path
    records = load_frame_records(records_path)

    by_camera = _records_by_camera(
        records,
        episode_id=episode_id,
        frame_count=manifest.frame_count,
    )

    transforms = {
        camera.camera: camera.transform
        for camera in manifest.cameras
    }
    if set(transforms) != {"front", "wrist"}:
        raise ValueError(
            f"{episode_id}: expected front/wrist camera manifests"
        )

    state_scores = build_state_scores(samples)
    front_motion = visual_motion_scores(
        episode_dir=episode_dir,
        records_by_frame=by_camera["front"],
        transform=transforms["front"],
        frame_count=manifest.frame_count,
        stride=selection_config.visual_stride,
        width=selection_config.visual_width,
        height=selection_config.visual_height,
    )
    wrist_motion = visual_motion_scores(
        episode_dir=episode_dir,
        records_by_frame=by_camera["wrist"],
        transform=transforms["wrist"],
        frame_count=manifest.frame_count,
        stride=selection_config.visual_stride,
        width=selection_config.visual_width,
        height=selection_config.visual_height,
    )

    selected = select_review_candidates(
        samples=samples,
        state_scores=state_scores,
        front_motion=front_motion,
        wrist_motion=wrist_motion,
        config=selection_config,
    )

    return [
        CandidateFrame(
            frame_index=item.frame_index,
            timestamp_sec=item.timestamp_sec,
            reasons=tuple(sorted(item.reasons)),
            metrics={
                key: float(value)
                for key, value in sorted(item.metrics.items())
            },
        )
        for item in selected
    ]


def build_day17_report(
    *,
    candidate_frames_by_episode: dict[str, list[CandidateFrame]],
    gold_events: list[GoldFailureEvent],
    tolerance_frames: Iterable[int],
    selector_config_path: str,
    selector_config_sha256: str,
) -> dict[str, Any]:
    tolerances = sorted({int(value) for value in tolerance_frames})
    verified_events = [
        event for event in gold_events if event.is_verified
    ]
    unresolved_events = [
        event
        for event in gold_events
        if event.review_disposition == "reviewed_unresolved"
    ]

    results: list[EventLocalizationResult] = []
    for event in sorted(
        verified_events,
        key=lambda item: item.event_id,
    ):
        candidates = candidate_frames_by_episode.get(
            event.episode_id
        )
        if candidates is None:
            raise ValueError(
                f"{event.episode_id}: missing candidate set"
            )
        results.append(
            evaluate_verified_event(
                event=event,
                candidates=candidates,
                tolerance_frames=tolerances,
            )
        )

    metrics = aggregate_localization_results(
        results=results,
        unresolved_events=unresolved_events,
        tolerance_frames=tolerances,
    )

    return {
        "schema_version": REPORT_SCHEMA,
        "benchmark_status": BENCHMARK_STATUS,
        "candidate_selection_status": SELECTOR_STATUS,
        "gold_read_during_candidate_generation": False,
        "selector_config_path": selector_config_path,
        "selector_config_sha256": selector_config_sha256,
        "evaluation_contract": {
            "unit": "selected_candidate_frame_vs_verified_failure_interval",
            "exact_hit_rule": (
                "at_least_one_selected_frame_inside_gold_interval"
            ),
            "tolerance_frames": tolerances,
            "unresolved_policy": (
                "reviewed_unresolved_events_are_excluded_from_"
                "supervised_interval_metrics_and_are_not_negatives"
            ),
            "boundary_iou": (
                "not_applicable_point_selector_does_not_predict_intervals"
            ),
            "quality_threshold": (
                "none_diagnostic_measurement_no_post_gt_tuning"
            ),
        },
        "candidate_sets": {
            episode_id: [
                candidate.to_dict()
                for candidate in candidates
            ]
            for episode_id, candidates in sorted(
                candidate_frames_by_episode.items()
            )
        },
        "event_results": [
            result.to_dict()
            for result in results
        ],
        "metrics": metrics,
    }
