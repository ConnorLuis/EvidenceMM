from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evidencemm.state_action_selection import load_state_action_samples


SCHEMA_VERSION = "day16_human_gt_v1"
HUMAN_REVIEWER = "day16_human_review"

REVIEW_VERIFIED = "verified"
REVIEW_UNRESOLVED = "reviewed_unresolved"

EXPECTED_EVENT_MODES = {
    "20260815_111613_event_01": "grasp_drop",
    "20260815_111613_event_02": "post_place_collision",
    "20260815_112058_event_01": "post_place_collision",
    "20260815_112633_event_01": "post_place_collision",
    "20260815_112859_event_01": "post_place_collision",
    "20260815_140119_event_01": "object_push_during_grasp",
    "20260815_141416_event_01": "drop_above_target",
    "20260815_141657_event_01": "post_place_collision",
    "20260815_155139_event_01": "grasp_drop",
}

EXPECTED_VERIFIED_COUNT = 7
EXPECTED_UNRESOLVED_COUNT = 2


class HumanGTValidationError(ValueError):
    pass


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanGTValidationError(
                f"{path}:{lineno}: invalid JSON: {exc}"
            ) from exc

        if not isinstance(value, dict):
            raise HumanGTValidationError(
                f"{path}:{lineno}: expected JSON object"
            )

        rows.append(value)

    return rows


def _notes_present(notes: Any) -> bool:
    if notes is None:
        return False

    if isinstance(notes, str):
        return bool(notes.strip())

    if isinstance(notes, (list, dict)):
        return bool(notes)

    return bool(str(notes).strip())


def classify_review_disposition(
    event: dict[str, Any],
    *,
    reviewer: str | None,
    notes: Any,
) -> str:
    status = event.get("event_status")

    if status == "verified":
        if event.get("failure_interval") is None:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "verified event requires failure_interval"
            )

        if event.get("causal_diagnosis") is None:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "verified event requires causal_diagnosis"
            )

        if event.get("confidence") is None:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "verified event requires confidence"
            )

        if not event.get("supporting_robot_refs"):
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "verified event requires supporting_robot_refs"
            )

        if reviewer != HUMAN_REVIEWER:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                f"unexpected reviewer={reviewer!r}"
            )

        return REVIEW_VERIFIED

    if status == "draft":
        if reviewer != HUMAN_REVIEWER:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "draft event has not completed human review"
            )

        if not _notes_present(notes):
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "reviewed unresolved event requires review notes"
            )

        if event.get("failure_interval") is not None:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "reviewed unresolved event must not invent "
                "failure_interval"
            )

        if event.get("causal_diagnosis") is not None:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "reviewed unresolved event must not invent "
                "causal_diagnosis"
            )

        if event.get("confidence") is not None:
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "reviewed unresolved event must not have confidence"
            )

        if event.get("supporting_robot_refs"):
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "reviewed unresolved event must not have "
                "supporting_robot_refs"
            )

        if event.get("counterevidence_robot_refs"):
            raise HumanGTValidationError(
                f"{event.get('event_id')}: "
                "reviewed unresolved event must not have "
                "counterevidence_robot_refs"
            )

        return REVIEW_UNRESOLVED

    raise HumanGTValidationError(
        f"{event.get('event_id')}: "
        f"unsupported event_status={status!r}"
    )


def _flatten_source_cases(
    source_cases_path: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for case in _load_jsonl(source_cases_path):
        episode_id = case.get("episode_id")

        if not episode_id:
            raise HumanGTValidationError(
                "source annotation case missing episode_id"
            )

        events = case.get("events")

        if not isinstance(events, list):
            raise HumanGTValidationError(
                f"{episode_id}: source case events must be a list"
            )

        for event in events:
            event_id = event.get("event_id")

            if not event_id:
                raise HumanGTValidationError(
                    f"{episode_id}: source event missing event_id"
                )

            if event_id in result:
                raise HumanGTValidationError(
                    f"duplicate source event_id={event_id}"
                )

            result[event_id] = {
                "episode_id": episode_id,
                "event": event,
                "case": case,
            }

    return result


def _load_review_templates(
    review_root: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    paths = sorted(
        review_root.glob("*/review_template.json")
    )

    if len(paths) != 8:
        raise HumanGTValidationError(
            "expected exactly 8 review templates, "
            f"found {len(paths)}"
        )

    for path in paths:
        case = _load_json(path)

        episode_id = case.get("episode_id")

        if not episode_id:
            raise HumanGTValidationError(
                f"{path}: missing episode_id"
            )

        events = case.get("events")

        if not isinstance(events, list):
            raise HumanGTValidationError(
                f"{path}: events must be a list"
            )

        for event in events:
            event_id = event.get("event_id")

            if not event_id:
                raise HumanGTValidationError(
                    f"{path}: event missing event_id"
                )

            if event_id in result:
                raise HumanGTValidationError(
                    f"duplicate review event_id={event_id}"
                )

            result[event_id] = {
                "episode_id": episode_id,
                "event": event,
                "case": case,
                "path": path,
            }

    return result


def _sample_map(
    dataset_root: Path,
    episode_id: str,
) -> dict[int, Any]:
    path = (
        dataset_root
        / episode_id
        / "samples.csv"
    )

    if not path.exists():
        raise HumanGTValidationError(
            f"{episode_id}: samples.csv not found: {path}"
        )

    samples = load_state_action_samples(
        path,
        verify_tracking_error=True,
    )

    return {
        int(sample.frame_index): sample
        for sample in samples
    }


def _assert_close(
    *,
    actual: float,
    expected: float,
    label: str,
    tolerance: float = 1e-5,
) -> None:
    if abs(actual - expected) > tolerance:
        raise HumanGTValidationError(
            f"{label}: actual={actual}, expected={expected}"
        )


def _validate_ref(
    *,
    ref: dict[str, Any],
    episode_id: str,
    samples: dict[int, Any],
    dataset_root: Path,
    interval: dict[str, Any] | None,
    require_interval_overlap: bool,
) -> None:
    if ref.get("source_id") != episode_id:
        raise HumanGTValidationError(
            f"{episode_id}: EvidenceRef source_id mismatch: "
            f"{ref.get('source_id')!r}"
        )

    if ref.get("source_type") != "robot_sequence":
        raise HumanGTValidationError(
            f"{episode_id}: unsupported EvidenceRef source_type="
            f"{ref.get('source_type')!r}"
        )

    camera = ref.get("camera")

    if camera not in {"front", "wrist"}:
        raise HumanGTValidationError(
            f"{episode_id}: invalid EvidenceRef camera={camera!r}"
        )

    frame = int(ref["frame_index"])

    if frame not in samples:
        raise HumanGTValidationError(
            f"{episode_id}: EvidenceRef frame {frame} "
            "not found in samples"
        )

    timestamp = float(
        samples[frame].timestamp_sec
    )

    _assert_close(
        actual=float(ref["time_start_sec"]),
        expected=timestamp,
        label=(
            f"{episode_id} frame={frame} "
            "EvidenceRef time_start_sec"
        ),
    )

    _assert_close(
        actual=float(ref["time_end_sec"]),
        expected=timestamp,
        label=(
            f"{episode_id} frame={frame} "
            "EvidenceRef time_end_sec"
        ),
    )

    image_path = (
        dataset_root
        / episode_id
        / camera
        / f"{frame:06d}.jpg"
    )

    if not image_path.exists():
        raise HumanGTValidationError(
            f"{episode_id}: EvidenceRef image not found: "
            f"{image_path}"
        )

    if require_interval_overlap and interval is not None:
        start_frame = int(
            interval["start_frame"]
        )
        end_frame = int(
            interval["end_frame"]
        )

        if not (
            start_frame
            <= frame
            <= end_frame
        ):
            raise HumanGTValidationError(
                f"{episode_id}: supporting EvidenceRef "
                f"frame={frame} lies outside "
                f"[{start_frame}, {end_frame}]"
            )


def _validate_verified_binding(
    *,
    episode_id: str,
    event: dict[str, Any],
    samples: dict[int, Any],
    dataset_root: Path,
) -> None:
    interval = event["failure_interval"]

    required_keys = {
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
    }

    if set(interval) != required_keys:
        raise HumanGTValidationError(
            f"{event['event_id']}: unexpected "
            f"failure_interval keys={sorted(interval)}"
        )

    start_frame = int(
        interval["start_frame"]
    )
    end_frame = int(
        interval["end_frame"]
    )

    if start_frame > end_frame:
        raise HumanGTValidationError(
            f"{event['event_id']}: "
            "start_frame > end_frame"
        )

    if (
        start_frame not in samples
        or end_frame not in samples
    ):
        raise HumanGTValidationError(
            f"{event['event_id']}: "
            "interval frame not found in samples"
        )

    start_sec = float(
        samples[start_frame].timestamp_sec
    )
    end_sec = float(
        samples[end_frame].timestamp_sec
    )

    _assert_close(
        actual=float(interval["start_sec"]),
        expected=start_sec,
        label=f"{event['event_id']} start_sec",
    )

    _assert_close(
        actual=float(interval["end_sec"]),
        expected=end_sec,
        label=f"{event['event_id']} end_sec",
    )

    supporting = event.get(
        "supporting_robot_refs",
        [],
    )

    for ref in supporting:
        _validate_ref(
            ref=ref,
            episode_id=episode_id,
            samples=samples,
            dataset_root=dataset_root,
            interval=interval,
            require_interval_overlap=True,
        )

    for ref in event.get(
        "counterevidence_robot_refs",
        [],
    ):
        _validate_ref(
            ref=ref,
            episode_id=episode_id,
            samples=samples,
            dataset_root=dataset_root,
            interval=interval,
            require_interval_overlap=False,
        )


def build_human_gt_records(
    *,
    review_root: Path,
    source_cases_path: Path,
    dataset_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    source = _flatten_source_cases(
        source_cases_path
    )

    reviews = _load_review_templates(
        review_root
    )

    expected_ids = set(
        EXPECTED_EVENT_MODES
    )

    if set(source) != expected_ids:
        raise HumanGTValidationError(
            "source event universe mismatch: "
            f"missing={sorted(expected_ids - set(source))}, "
            f"extra={sorted(set(source) - expected_ids)}"
        )

    if set(reviews) != expected_ids:
        raise HumanGTValidationError(
            "review event universe mismatch: "
            f"missing={sorted(expected_ids - set(reviews))}, "
            f"extra={sorted(set(reviews) - expected_ids)}"
        )

    sample_cache: dict[
        str,
        dict[int, Any],
    ] = {}

    records: list[dict[str, Any]] = []

    for event_id in sorted(expected_ids):
        expected_mode = (
            EXPECTED_EVENT_MODES[event_id]
        )

        source_item = source[event_id]
        review_item = reviews[event_id]

        if (
            source_item["episode_id"]
            != review_item["episode_id"]
        ):
            raise HumanGTValidationError(
                f"{event_id}: episode_id mismatch "
                "between source and review"
            )

        episode_id = review_item[
            "episode_id"
        ]

        source_event = source_item["event"]
        event = review_item["event"]
        case = review_item["case"]

        if (
            source_event.get(
                "observed_failure_mode"
            )
            != expected_mode
        ):
            raise HumanGTValidationError(
                f"{event_id}: source mode mismatch"
            )

        if (
            event.get(
                "observed_failure_mode"
            )
            != expected_mode
        ):
            raise HumanGTValidationError(
                f"{event_id}: review mode mismatch"
            )

        reviewer = (
            case.get("reviewer")
            or ""
        )

        notes = case.get("notes")

        disposition = (
            classify_review_disposition(
                event,
                reviewer=reviewer,
                notes=notes,
            )
        )

        if episode_id not in sample_cache:
            sample_cache[episode_id] = (
                _sample_map(
                    dataset_root,
                    episode_id,
                )
            )

        samples = sample_cache[
            episode_id
        ]

        if disposition == REVIEW_VERIFIED:
            _validate_verified_binding(
                episode_id=episode_id,
                event=event,
                samples=samples,
                dataset_root=dataset_root,
            )

        episode_context = {
            key: value
            for key, value in case.items()
            if key not in {
                "events",
                "reviewer",
                "notes",
            }
        }

        record = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            **event,
            "review_disposition": disposition,
            "reviewer": reviewer,
            "review_notes": notes,
            "episode_context": episode_context,
            "source_review_template": (
                review_item["path"].as_posix()
            ),
        }

        records.append(record)

    counts = Counter(
        record["review_disposition"]
        for record in records
    )

    verified_count = counts[
        REVIEW_VERIFIED
    ]

    unresolved_count = counts[
        REVIEW_UNRESOLVED
    ]

    if (
        verified_count
        != EXPECTED_VERIFIED_COUNT
    ):
        raise HumanGTValidationError(
            "expected "
            f"{EXPECTED_VERIFIED_COUNT} verified events, "
            f"found {verified_count}"
        )

    if (
        unresolved_count
        != EXPECTED_UNRESOLVED_COUNT
    ):
        raise HumanGTValidationError(
            "expected "
            f"{EXPECTED_UNRESOLVED_COUNT} "
            "reviewed-unresolved events, "
            f"found {unresolved_count}"
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len({
            record["episode_id"]
            for record in records
        }),
        "event_count": len(records),
        "verified_count": verified_count,
        "reviewed_unresolved_count": (
            unresolved_count
        ),
        "waiting_for_review_count": 0,
        "human_review_complete": (
            len(records)
            == (
                verified_count
                + unresolved_count
            )
        ),
        "verified_event_ids": [
            record["event_id"]
            for record in records
            if (
                record["review_disposition"]
                == REVIEW_VERIFIED
            )
        ],
        "reviewed_unresolved_event_ids": [
            record["event_id"]
            for record in records
            if (
                record["review_disposition"]
                == REVIEW_UNRESOLVED
            )
        ],
    }

    return records, summary


def write_human_gt(
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
    summary_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_promoted_human_gt(
    *,
    review_root: Path,
    source_cases_path: Path,
    dataset_root: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    expected_records, expected_summary = (
        build_human_gt_records(
            review_root=review_root,
            source_cases_path=source_cases_path,
            dataset_root=dataset_root,
        )
    )

    actual_records = _load_jsonl(
        output_path
    )

    actual_summary = _load_json(
        summary_path
    )

    if actual_records != expected_records:
        raise HumanGTValidationError(
            "promoted human-GT JSONL does not "
            "match current reviewed templates"
        )

    if actual_summary != expected_summary:
        raise HumanGTValidationError(
            "human-GT summary does not match "
            "current reviewed templates"
        )

    return actual_summary
