from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CONFIG_SCHEMA = (
    "evidencemm_day20_heldout_interval_eval_config_v1"
)
REPORT_SCHEMA = (
    "evidencemm_day20_heldout_interval_eval_report_v1"
)
EVALUATION_STATUS = (
    "prospective_procedural_heldout_interval_evaluation"
)

DAY19_MODEL_SCHEMA = (
    "evidencemm_day19_interval_localizer_model_v1"
)
DAY19_MODEL_STATUS = (
    "development_selected_interval_proposal_localizer"
)

VERIFIED = "verified"
REVIEWED_UNRESOLVED = "reviewed_unresolved"


@dataclass(frozen=True)
class HeldoutGoldEvent:
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
class HeldoutEventResult:
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


def load_heldout_gold_events(
    path: str | Path,
    *,
    allowed_episode_ids: set[str],
) -> list[HeldoutGoldEvent]:
    source = Path(path)
    events: list[HeldoutGoldEvent] = []
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

        episode_id = str(
            row.get("episode_id", "")
        )

        # Critical Day20 barrier:
        # skip every non-held-out row BEFORE materializing
        # its failure_interval.
        if episode_id not in allowed_episode_ids:
            continue

        event_id = str(
            row.get("event_id", "")
        )
        if not event_id:
            raise ValueError(
                f"{source}:{lineno}: missing event_id"
            )
        if event_id in seen:
            raise ValueError(
                f"duplicate held-out event_id={event_id}"
            )
        seen.add(event_id)

        disposition = str(
            row.get("review_disposition", "")
        )

        if disposition == VERIFIED:
            interval = row.get(
                "failure_interval"
            )
            if not isinstance(interval, dict):
                raise ValueError(
                    f"{event_id}: verified held-out "
                    "event requires failure_interval"
                )
            start_frame = int(
                interval["start_frame"]
            )
            end_frame = int(
                interval["end_frame"]
            )
            if start_frame > end_frame:
                raise ValueError(
                    f"{event_id}: invalid failure interval"
                )
        elif disposition == REVIEWED_UNRESOLVED:
            if row.get(
                "failure_interval"
            ) is not None:
                raise ValueError(
                    f"{event_id}: unresolved held-out event "
                    "must not have failure_interval"
                )
            start_frame = None
            end_frame = None
        else:
            raise ValueError(
                f"{event_id}: unsupported "
                "review_disposition={disposition!r}"
            )

        events.append(
            HeldoutGoldEvent(
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


def validate_expected_heldout_gold(
    events: Sequence[HeldoutGoldEvent],
    *,
    expected_event_count: int,
    expected_verified_count: int,
    expected_unresolved_count: int,
    expected_event_ids: Sequence[str],
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
    actual_ids = [
        event.event_id
        for event in events
    ]
    wanted_ids = sorted(
        str(value)
        for value in expected_event_ids
    )

    if len(events) != int(
        expected_event_count
    ):
        raise ValueError(
            "held-out event_count mismatch"
        )
    if len(verified) != int(
        expected_verified_count
    ):
        raise ValueError(
            "held-out verified_event_count mismatch"
        )
    if len(unresolved) != int(
        expected_unresolved_count
    ):
        raise ValueError(
            "held-out reviewed_unresolved_count mismatch"
        )
    if actual_ids != wanted_ids:
        raise ValueError(
            f"held-out event IDs mismatch: "
            f"expected={wanted_ids}, actual={actual_ids}"
        )


def validate_frozen_day19_model(
    model: dict[str, Any],
    *,
    expected_selected_radius_frames: int,
    split_sha256: str,
    selector_config_sha256: str,
    human_gt_sha256: str,
) -> int:
    if model.get(
        "schema_version"
    ) != DAY19_MODEL_SCHEMA:
        raise ValueError(
            "unexpected Day19 model schema_version"
        )
    if model.get(
        "model_status"
    ) != DAY19_MODEL_STATUS:
        raise ValueError(
            "unexpected Day19 model_status"
        )
    if model.get(
        "model_selection_split"
    ) != "development":
        raise ValueError(
            "Day19 model must be development-selected"
        )

    anti = model.get(
        "anti_leakage",
        {},
    )
    if anti.get(
        "held_out_gt_used_for_model_selection"
    ) is not False:
        raise ValueError(
            "Day19 model does not preserve held-out barrier"
        )
    if anti.get(
        "held_out_metrics_reported"
    ) is not False:
        raise ValueError(
            "Day19 unexpectedly contains held-out metrics"
        )

    provenance = model.get(
        "provenance",
        {},
    )
    expected_hashes = {
        "benchmark_split_sha256": (
            split_sha256
        ),
        "frozen_selector_config_sha256": (
            selector_config_sha256
        ),
        "human_gt_sha256": human_gt_sha256,
    }
    for key, expected in expected_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"Day19 provenance mismatch for {key}"
            )

    radius = int(
        model["localizer"][
            "selected_radius_frames"
        ]
    )
    if radius != int(
        expected_selected_radius_frames
    ):
        raise ValueError(
            "Day19 selected radius differs from "
            "Day20 frozen expectation"
        )
    return radius


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
    return intersection / (
        size_a + size_b - intersection
    )


def evaluate_verified_event(
    *,
    event: HeldoutGoldEvent,
    proposals: Sequence[Any],
) -> HeldoutEventResult:
    if not event.is_verified:
        raise ValueError(
            f"{event.event_id}: only verified "
            "held-out events may be scored"
        )
    if (
        event.start_frame is None
        or event.end_frame is None
    ):
        raise ValueError(
            f"{event.event_id}: missing verified interval"
        )
    if not proposals:
        raise ValueError(
            f"{event.episode_id}: empty proposal list"
        )

    ranked = sorted(
        (
            (
                interval_iou(
                    int(proposal.start_frame),
                    int(proposal.end_frame),
                    event.start_frame,
                    event.end_frame,
                ),
                abs(
                    int(proposal.center_frame)
                    - (
                        event.start_frame
                        + event.end_frame
                    )
                    / 2.0
                ),
                int(proposal.center_frame),
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

    return HeldoutEventResult(
        event_id=event.event_id,
        episode_id=event.episode_id,
        observed_failure_mode=(
            event.observed_failure_mode
        ),
        gold_start_frame=event.start_frame,
        gold_end_frame=event.end_frame,
        best_proposal_center_frame=int(
            best.center_frame
        ),
        best_proposal_start_frame=int(
            best.start_frame
        ),
        best_proposal_end_frame=int(
            best.end_frame
        ),
        best_iou=float(
            best_iou
        ),
        overlap_hit=(best_iou > 0.0),
        onset_abs_error_frames=abs(
            int(best.start_frame)
            - event.start_frame
        ),
        offset_abs_error_frames=abs(
            int(best.end_frame)
            - event.end_frame
        ),
    )


def _threshold_key(
    value: float,
) -> str:
    return f"{float(value):.2f}"


def metric_block(
    results: Sequence[HeldoutEventResult],
    *,
    iou_thresholds: Sequence[float],
) -> dict[str, Any]:
    thresholds = sorted(
        float(value)
        for value in iou_thresholds
    )
    if any(
        value <= 0.0 or value > 1.0
        for value in thresholds
    ):
        raise ValueError(
            "IoU thresholds must lie in (0, 1]"
        )

    if not results:
        return {
            "event_count": 0,
            "event_recall": None,
            "mean_best_iou": None,
            "median_best_iou": None,
            "recall_at_iou": {
                _threshold_key(value): None
                for value in thresholds
            },
            "mean_onset_abs_error_frames": None,
            "mean_offset_abs_error_frames": None,
        }

    count = len(results)
    return {
        "event_count": count,
        "event_recall": (
            sum(
                result.overlap_hit
                for result in results
            )
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
            for threshold in thresholds
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


def aggregate_metrics(
    results: Sequence[HeldoutEventResult],
    *,
    iou_thresholds: Sequence[float],
) -> dict[str, Any]:
    by_mode: dict[
        str,
        list[HeldoutEventResult],
    ] = {}
    for result in results:
        by_mode.setdefault(
            result.observed_failure_mode,
            [],
        ).append(result)

    return {
        "overall": metric_block(
            results,
            iou_thresholds=iou_thresholds,
        ),
        "by_observed_failure_mode": {
            mode: metric_block(
                mode_results,
                iou_thresholds=iou_thresholds,
            )
            for mode, mode_results in sorted(
                by_mode.items()
            )
        },
    }


def build_report(
    *,
    frozen_after_day19_commit: str,
    day19_model_blob_sha1: str,
    split_sha256: str,
    model_sha256: str,
    selector_config_sha256: str,
    human_gt_sha256: str,
    selected_radius_frames: int,
    proposals_by_episode: dict[
        str,
        Sequence[Any],
    ],
    gold_events: Sequence[HeldoutGoldEvent],
    iou_thresholds: Sequence[float],
) -> dict[str, Any]:
    verified = [
        event
        for event in gold_events
        if event.is_verified
    ]
    unresolved = [
        event
        for event in gold_events
        if not event.is_verified
    ]

    results = []
    for event in verified:
        proposals = proposals_by_episode.get(
            event.episode_id
        )
        if proposals is None:
            raise ValueError(
                f"{event.event_id}: missing held-out proposals"
            )
        results.append(
            evaluate_verified_event(
                event=event,
                proposals=proposals,
            )
        )

    proposal_payload = {
        episode_id: [
            {
                "center_frame": int(
                    proposal.center_frame
                ),
                "start_frame": int(
                    proposal.start_frame
                ),
                "end_frame": int(
                    proposal.end_frame
                ),
                "radius_frames": int(
                    proposal.radius_frames
                ),
                "center_timestamp_sec": float(
                    proposal.center_timestamp_sec
                ),
                "reasons": [
                    str(reason)
                    for reason in proposal.reasons
                ],
            }
            for proposal in proposals
        ]
        for episode_id, proposals in sorted(
            proposals_by_episode.items()
        )
    }

    return {
        "schema_version": REPORT_SCHEMA,
        "evaluation_status": EVALUATION_STATUS,
        "evaluation_split": "held_out",
        "provenance": {
            "frozen_after_day19_commit": (
                frozen_after_day19_commit
            ),
            "day19_model_blob_sha1": (
                day19_model_blob_sha1
            ),
            "benchmark_split_sha256": (
                split_sha256
            ),
            "interval_model_sha256": (
                model_sha256
            ),
            "frozen_selector_config_sha256": (
                selector_config_sha256
            ),
            "human_gt_sha256": human_gt_sha256,
        },
        "anti_leakage": {
            "candidate_generation_uses_gold": False,
            "held_out_gt_loaded_after_candidate_generation": True,
            "model_selection_performed": False,
            "radius_tuned_on_held_out": False,
            "post_heldout_tuning_allowed": False,
            "day18_split_changed": False,
            "day19_model_changed": False,
            "note": (
                "This is the one frozen-model held-out evaluation. "
                "The Day18 split was created after Human GT existed, "
                "so the claim is procedural/prospective relative to "
                "Day19 model selection, not a pristine unseen-human "
                "external benchmark."
            ),
        },
        "localizer": {
            "selected_radius_frames": int(
                selected_radius_frames
            ),
            "center_policy": (
                "frozen_day16_signal_candidates_excluding_uniform_only"
            ),
            "interval_rule": (
                "symmetric_radius_around_candidate_center"
            ),
        },
        "held_out_universe": {
            "anomaly_episode_ids": sorted(
                proposals_by_episode
            ),
            "verified_event_count": len(
                verified
            ),
            "reviewed_unresolved_count": len(
                unresolved
            ),
            "reviewed_unresolved_event_ids": sorted(
                event.event_id
                for event in unresolved
            ),
        },
        "held_out_proposals": proposal_payload,
        "event_results": [
            result.to_dict()
            for result in results
        ],
        "metrics": aggregate_metrics(
            results,
            iou_thresholds=iou_thresholds,
        ),
        "evaluation_seal": {
            "frozen_model_final_evaluation": True,
            "same_held_out_set_may_be_reused_for_diagnostics": True,
            "same_held_out_set_may_be_used_for_future_model_selection": False,
            "new_prospective_claim_requires_new_held_out_data_or_split": True,
        },
    }
