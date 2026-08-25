#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import webbrowser
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    ROOT
    / "data/protocol/day29_pass_a_operational_contract.json"
)

POPULATION_PATH = (
    ROOT
    / "data/annotations/day29_blind_review_records.jsonl"
)

RECORDS_PATH = (
    ROOT
    / "data/annotations/day29_pass_a_records.jsonl"
)

FREEZE_RECEIPT_PATH = (
    ROOT
    / "data/protocol/day29_pass_a_freeze_receipt.json"
)

PACK_ROOT = (
    ROOT
    / "reports/day29_blind_review_pack"
)

PACK_MANIFEST_PATH = PACK_ROOT / "manifest.json"

MATERIALIZATION_RECEIPT_PATH = (
    ROOT
    / "data/protocol/day29_blind_review_pack_materialization.json"
)

MANUAL_COVERAGE_PATH = (
    ROOT
    / "data/protocol/day29_manual_evidence_coverage.json"
)

MANUAL_POLICY_PATH = (
    ROOT
    / "docs/day29_manual_evidence_policy_amendment_v1.md"
)

PASS_A_SCHEMA = "evidencemm_day29_pass_a_record_v1"

CAUSES = {
    "target_offset_or_perception",
    "gripper_close_timing",
    "trajectory_execution_deviation",
}

ANSWERABILITY = {
    "answerable",
    "insufficient_evidence",
    "not_applicable_clean",
}

FORBIDDEN_RECORD_FIELDS = {
    "pair_group_id",
    "plan_row_id",
    "planned_physical_cause",
    "intervention_type",
    "intervention_parameters",
    "intervention_applied",
    "technical_valid",
    "experimental_valid",
    "task_success",
    "intervention_verified",
    "physical_cause_gt",
    "diagnostic_decision_gt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def require_ancestor(commit: str) -> None:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"required frozen commit is not an ancestor: {commit}"
        )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def tree_records(
    directory: Path,
) -> list[str]:
    records: list[str] = []

    def visit(current: Path) -> None:
        for entry in sorted(
            current.iterdir(),
            key=lambda p: p.name,
        ):
            rel = entry.relative_to(
                directory
            ).as_posix()

            if entry.is_symlink():
                records.append(
                    "L\t"
                    + rel
                    + "\t"
                    + os.readlink(entry)
                )
                continue

            if entry.is_dir():
                visit(entry)
                continue

            if entry.is_file():
                records.append(
                    "F\t"
                    + rel
                    + "\t"
                    + sha256_file(entry)
                )
                continue

            raise RuntimeError(
                f"unsupported pack entry: {entry}"
            )

    visit(directory)

    return records


def pack_tree_sha256() -> str:
    payload = (
        "\n".join(tree_records(PACK_ROOT))
        + "\n"
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def verify_frozen_environment() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    contract = read_json(CONTRACT_PATH)

    deps = contract["frozen_dependencies"]

    require_ancestor(
        deps["manual_policy_commit"]
    )

    if sha256_file(POPULATION_PATH) != deps[
        "blank_population_sha256"
    ]:
        raise RuntimeError(
            "blank population SHA mismatch"
        )

    if sha256_file(MANUAL_COVERAGE_PATH) != deps[
        "manual_coverage_sha256"
    ]:
        raise RuntimeError(
            "manual coverage SHA mismatch"
        )

    if sha256_file(MANUAL_POLICY_PATH) != deps[
        "manual_policy_amendment_sha256"
    ]:
        raise RuntimeError(
            "manual policy SHA mismatch"
        )

    materialization = read_json(
        MATERIALIZATION_RECEIPT_PATH
    )

    if materialization[
        "pack_tree_sha256"
    ] != deps["pack_tree_sha256"]:
        raise RuntimeError(
            "frozen receipt pack-tree SHA mismatch"
        )

    actual_tree = pack_tree_sha256()

    if actual_tree != deps["pack_tree_sha256"]:
        raise RuntimeError(
            "official blind pack has changed"
        )

    manifest = read_json(
        PACK_MANIFEST_PATH
    )

    if manifest["case_count"] != 90:
        raise RuntimeError(
            "unexpected pack case count"
        )

    if manifest[
        "review_order_sha256"
    ] != deps["review_order_sha256"]:
        raise RuntimeError(
            "review-order SHA mismatch"
        )

    if manifest["human_review_started"] is not False:
        raise RuntimeError(
            "pack manifest human_review_started changed"
        )

    if manifest[
        "future_split_materialized"
    ] is not False:
        raise RuntimeError(
            "future split unexpectedly materialized"
        )

    return contract, manifest


def frame_rows(
    episode_id: str,
) -> list[dict[str, Any]]:
    path = (
        PACK_ROOT
        / episode_id
        / "full_state_action.json"
    )

    payload = read_json(path)

    rows = payload["rows"]

    if len(rows) != 900:
        raise RuntimeError(
            f"{episode_id}: expected 900 state rows"
        )

    return rows


def timestamp_for(
    episode_id: str,
    frame_index: int,
) -> float:
    rows = frame_rows(episode_id)

    if not 0 <= frame_index < len(rows):
        raise ValueError(
            f"frame index out of range: {frame_index}"
        )

    row = rows[frame_index]

    if row["frame_index"] != frame_index:
        raise ValueError(
            "non-contiguous full-state rows"
        )

    return float(row["timestamp_sec"])


def blank_pass_a_record(
    review_position: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PASS_A_SCHEMA,
        "review_position": review_position,
        "episode_id": source["episode_id"],
        "observed_symptom": None,
        "failure_interval": None,
        "supporting_robot_refs": [],
        "counterevidence_robot_refs": [],
        "supporting_manual_refs": [],
        "counterevidence_manual_refs": [],
        "evidence_answerability_gt": None,
        "explicit_uncertainty_reason": None,
        "blind_confidence": None,
        "blind_review_notes": None,
        "blind_cause_hypothesis": None,
    }


def init_records() -> None:
    _, manifest = verify_frozen_environment()

    if RECORDS_PATH.exists():
        raise RuntimeError(
            "Pass A records already exist"
        )

    population = read_jsonl(
        POPULATION_PATH
    )

    by_id = {
        row["episode_id"]: row
        for row in population
    }

    if len(by_id) != 90:
        raise RuntimeError(
            "unexpected blind population count"
        )

    ordered_ids = [
        case["episode_id"]
        for case in manifest["cases"]
    ]

    if set(ordered_ids) != set(by_id):
        raise RuntimeError(
            "pack/population episode set mismatch"
        )

    rows = [
        blank_pass_a_record(
            position,
            by_id[episode_id],
        )
        for position, episode_id in enumerate(
            ordered_ids,
            start=1,
        )
    ]

    write_jsonl(
        RECORDS_PATH,
        rows,
    )

    print("records_created =", len(rows))
    print("PASS A INIT: PASS")


def ask(
    prompt: str,
    default: str | None = None,
) -> str:
    suffix = ""

    if default not in (None, ""):
        suffix = f" [{default}]"

    value = input(
        f"{prompt}{suffix}: "
    ).strip()

    if not value and default is not None:
        return default

    return value


def choose(
    prompt: str,
    allowed: list[str],
    default: str | None = None,
) -> str:
    while True:
        print()
        print(prompt)

        for index, value in enumerate(
            allowed,
            start=1,
        ):
            print(f"  {index}. {value}")

        raw = ask(
            "choice",
            default,
        )

        if raw in allowed:
            return raw

        try:
            numeric = int(raw)
        except ValueError:
            numeric = -1

        if 1 <= numeric <= len(allowed):
            return allowed[numeric - 1]

        print("invalid choice")


def robot_refs_to_text(
    refs: list[dict[str, Any]],
) -> str:
    parts: list[str] = []

    for ref in refs:
        camera = ref.get(
            "camera",
            "state",
        )

        if camera is None:
            camera = "state"

        parts.append(
            f"{camera}@{ref['frame_index']}:"
            f"{ref.get('note', '')}"
        )

    return ";".join(parts)


def parse_robot_refs(
    raw: str,
    episode_id: str,
) -> list[dict[str, Any]]:
    raw = raw.strip()

    if not raw:
        return []

    result: list[dict[str, Any]] = []

    for chunk in raw.split(";"):
        chunk = chunk.strip()

        if not chunk:
            continue

        if ":" not in chunk:
            raise ValueError(
                "robot ref requires :note"
            )

        locator, note = chunk.split(
            ":",
            1,
        )

        note = note.strip()

        if not note:
            raise ValueError(
                "robot ref note is required"
            )

        if "@" not in locator:
            raise ValueError(
                "robot ref requires channel@frame"
            )

        channel, frame_text = (
            locator.strip().split(
                "@",
                1,
            )
        )

        channel = channel.strip()

        if channel not in {
            "front",
            "wrist",
            "state",
        }:
            raise ValueError(
                f"invalid robot channel: {channel}"
            )

        frame_index = int(
            frame_text.strip()
        )

        timestamp = timestamp_for(
            episode_id,
            frame_index,
        )

        ref = {
            "source_id": episode_id,
            "source_type": "robot_sequence",
            "frame_index": frame_index,
            "time_start_sec": timestamp,
            "time_end_sec": timestamp,
            "note": note,
        }

        if channel in {
            "front",
            "wrist",
        }:
            ref["camera"] = channel

        result.append(ref)

    return result


def manual_refs_to_text(
    refs: list[dict[str, Any]],
) -> str:
    return ";".join(
        f"{ref['page_number']}:"
        f"{ref.get('relevance_note', '')}"
        for ref in refs
    )


def parse_manual_refs(
    raw: str,
) -> list[dict[str, Any]]:
    raw = raw.strip()

    if not raw:
        return []

    result: list[dict[str, Any]] = []

    for chunk in raw.split(";"):
        chunk = chunk.strip()

        if not chunk:
            continue

        if ":" not in chunk:
            raise ValueError(
                "manual ref requires page:relevance"
            )

        page_text, relevance = chunk.split(
            ":",
            1,
        )

        page = int(page_text.strip())
        relevance = relevance.strip()

        if page not in range(1, 9):
            raise ValueError(
                f"manual page out of range: {page}"
            )

        if not relevance:
            raise ValueError(
                "manual relevance note is required"
            )

        result.append(
            {
                "source_id": "sts3215_datasheet",
                "page_number": page,
                "relevance_note": relevance,
            }
        )

    return result


def interval_from_frames(
    episode_id: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    if not 0 <= start <= end <= 899:
        raise ValueError(
            "failure interval must satisfy "
            "0 <= start <= end <= 899"
        )

    return {
        "frame_start": start,
        "frame_end": end,
        "time_start_sec": timestamp_for(
            episode_id,
            start,
        ),
        "time_end_sec": timestamp_for(
            episode_id,
            end,
        ),
    }


def record_errors(
    record: dict[str, Any],
    require_complete: bool,
) -> list[str]:
    errors: list[str] = []

    episode_id = record.get(
        "episode_id",
        "",
    )

    unexpected = (
        set(record)
        & FORBIDDEN_RECORD_FIELDS
    )

    if unexpected:
        errors.append(
            "forbidden fields present: "
            + ",".join(sorted(unexpected))
        )

    if record.get(
        "schema_version"
    ) != PASS_A_SCHEMA:
        errors.append(
            "schema_version mismatch"
        )

    position = record.get(
        "review_position"
    )

    if not isinstance(position, int):
        errors.append(
            "review_position must be int"
        )

    symptom = record.get(
        "observed_symptom"
    )

    answerability = record.get(
        "evidence_answerability_gt"
    )

    cause = record.get(
        "blind_cause_hypothesis"
    )

    confidence = record.get(
        "blind_confidence"
    )

    if not require_complete and (
        symptom is None
        and answerability is None
        and confidence is None
    ):
        return errors

    if not isinstance(symptom, str) or not symptom.strip():
        errors.append(
            "observed_symptom required"
        )

    if answerability not in ANSWERABILITY:
        errors.append(
            "invalid evidence_answerability_gt"
        )
        return errors

    if not isinstance(
        confidence,
        (int, float),
    ) or isinstance(confidence, bool):
        errors.append(
            "blind_confidence must be numeric"
        )
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append(
            "blind_confidence outside [0,1]"
        )

    interval = record.get(
        "failure_interval"
    )

    if answerability == "not_applicable_clean":
        if interval is not None:
            errors.append(
                "clean record must have null failure_interval"
            )

        if cause is not None:
            errors.append(
                "clean record must have null blind cause"
            )

    else:
        if not isinstance(interval, dict):
            errors.append(
                "failed record requires failure_interval"
            )
        else:
            start = interval.get(
                "frame_start"
            )
            end = interval.get(
                "frame_end"
            )

            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= 899
            ):
                errors.append(
                    "invalid failure interval frames"
                )
            else:
                expected_start = timestamp_for(
                    episode_id,
                    start,
                )
                expected_end = timestamp_for(
                    episode_id,
                    end,
                )

                if abs(
                    float(
                        interval.get(
                            "time_start_sec",
                            -1,
                        )
                    )
                    - expected_start
                ) > 1e-6:
                    errors.append(
                        "failure interval start time mismatch"
                    )

                if abs(
                    float(
                        interval.get(
                            "time_end_sec",
                            -1,
                        )
                    )
                    - expected_end
                ) > 1e-6:
                    errors.append(
                        "failure interval end time mismatch"
                    )

    supporting_robot = record.get(
        "supporting_robot_refs",
        [],
    )

    if not isinstance(
        supporting_robot,
        list,
    ):
        errors.append(
            "supporting_robot_refs must be list"
        )
        supporting_robot = []

    for field in (
        "supporting_robot_refs",
        "counterevidence_robot_refs",
    ):
        refs = record.get(field, [])

        if not isinstance(refs, list):
            errors.append(
                f"{field} must be list"
            )
            continue

        for ref in refs:
            if ref.get(
                "source_id"
            ) != episode_id:
                errors.append(
                    f"{field}: source_id mismatch"
                )

            frame = ref.get(
                "frame_index"
            )

            if not isinstance(
                frame,
                int,
            ) or not 0 <= frame <= 899:
                errors.append(
                    f"{field}: invalid frame"
                )

            if not str(
                ref.get(
                    "note",
                    "",
                )
            ).strip():
                errors.append(
                    f"{field}: note required"
                )

    for field in (
        "supporting_manual_refs",
        "counterevidence_manual_refs",
    ):
        refs = record.get(field, [])

        if not isinstance(refs, list):
            errors.append(
                f"{field} must be list"
            )
            continue

        for ref in refs:
            if ref.get(
                "source_id"
            ) != "sts3215_datasheet":
                errors.append(
                    f"{field}: invalid source"
                )

            page = ref.get(
                "page_number"
            )

            if not isinstance(
                page,
                int,
            ) or not 1 <= page <= 8:
                errors.append(
                    f"{field}: invalid page"
                )

            if not str(
                ref.get(
                    "relevance_note",
                    "",
                )
            ).strip():
                errors.append(
                    f"{field}: relevance note required"
                )

    supporting_manual = record.get(
        "supporting_manual_refs",
        [],
    )

    notes = str(
        record.get(
            "blind_review_notes",
            "",
        )
        or ""
    )

    if answerability == "answerable":
        if cause not in CAUSES:
            errors.append(
                "answerable requires unique blind cause"
            )

        if not supporting_robot:
            errors.append(
                "answerable requires supporting robot refs"
            )

        if cause == "target_offset_or_perception":
            if supporting_manual:
                errors.append(
                    "target/perception supporting manual refs "
                    "must be empty"
                )

            if (
                "manual_support_not_applicable_to_claim"
                not in notes
            ):
                errors.append(
                    "target/perception answerable record "
                    "requires manual-not-applicable note"
                )

        elif cause in {
            "gripper_close_timing",
            "trajectory_execution_deviation",
        }:
            for ref in supporting_manual:
                if ref.get(
                    "page_number"
                ) not in {3, 4, 8}:
                    errors.append(
                        "supporting manual page must be "
                        "3, 4, or 8 for actuator-related claim"
                    )

            if (
                not supporting_manual
                and
                "manual_support_not_applicable_to_claim"
                not in notes
            ):
                errors.append(
                    "empty supporting manual refs require "
                    "manual-not-applicable note"
                )

    if answerability == "insufficient_evidence":
        if cause is not None:
            errors.append(
                "insufficient evidence requires null blind cause"
            )

        uncertainty = record.get(
            "explicit_uncertainty_reason"
        )

        if (
            not isinstance(
                uncertainty,
                str,
            )
            or not uncertainty.strip()
        ):
            errors.append(
                "insufficient evidence requires "
                "explicit_uncertainty_reason"
            )

    if answerability == "not_applicable_clean":
        if record.get(
            "explicit_uncertainty_reason"
        ) not in (
            None,
            "",
        ):
            errors.append(
                "clean record must not have uncertainty reason"
            )

    return errors


def is_complete(
    record: dict[str, Any],
) -> bool:
    return not record_errors(
        record,
        require_complete=True,
    )


def load_records() -> list[dict[str, Any]]:
    if not RECORDS_PATH.exists():
        raise RuntimeError(
            "Pass A records do not exist; run init"
        )

    return read_jsonl(
        RECORDS_PATH
    )


def save_records(
    records: list[dict[str, Any]],
) -> None:
    write_jsonl(
        RECORDS_PATH,
        records,
    )


def review_case(
    position: int | None,
) -> None:
    _, manifest = verify_frozen_environment()

    records = load_records()

    if position is None:
        candidate = next(
            (
                row
                for row in records
                if not is_complete(row)
            ),
            None,
        )

        if candidate is None:
            print(
                "All 90 Pass A records are complete."
            )
            return

        position = candidate[
            "review_position"
        ]

    if not 1 <= position <= len(records):
        raise ValueError(
            "review position out of range"
        )

    record = records[
        position - 1
    ]

    episode_id = record[
        "episode_id"
    ]

    expected_id = manifest[
        "cases"
    ][
        position - 1
    ][
        "episode_id"
    ]

    if episode_id != expected_id:
        raise RuntimeError(
            "review-order mismatch"
        )

    case_dir = (
        PACK_ROOT
        / episode_id
    )

    print()
    print("=" * 72)
    print(
        f"PASS A REVIEW "
        f"{position:02d}/90"
    )
    print("episode_id =", episode_id)
    print("=" * 72)
    print()
    print(
        "Opening blind evidence only."
    )

    webbrowser.open(
        (
            case_dir
            / "review.html"
        ).resolve().as_uri()
    )

    webbrowser.open(
        (
            case_dir
            / "frame_explorer.html"
        ).resolve().as_uri()
    )

    webbrowser.open(
        (
            PACK_ROOT
            / "manual/index.html"
        ).resolve().as_uri()
    )

    symptom = ask(
        "observed symptom",
        record.get(
            "observed_symptom"
        )
        or None,
    )

    answerability = choose(
        "evidence answerability",
        [
            "answerable",
            "insufficient_evidence",
            "not_applicable_clean",
        ],
        record.get(
            "evidence_answerability_gt"
        ),
    )

    cause: str | None = None
    interval: dict[str, Any] | None = None
    uncertainty: str | None = None

    if answerability == "answerable":
        cause = choose(
            "blind physical-cause hypothesis",
            [
                "target_offset_or_perception",
                "gripper_close_timing",
                "trajectory_execution_deviation",
            ],
            record.get(
                "blind_cause_hypothesis"
            ),
        )

    if answerability != "not_applicable_clean":
        existing_interval = (
            record.get(
                "failure_interval"
            )
            or {}
        )

        start_text = ask(
            "failure frame start",
            (
                str(
                    existing_interval[
                        "frame_start"
                    ]
                )
                if "frame_start"
                in existing_interval
                else None
            ),
        )

        end_text = ask(
            "failure frame end",
            (
                str(
                    existing_interval[
                        "frame_end"
                    ]
                )
                if "frame_end"
                in existing_interval
                else None
            ),
        )

        interval = interval_from_frames(
            episode_id,
            int(start_text),
            int(end_text),
        )

    print()
    print(
        "Robot ref syntax: "
        "front@412:note;"
        "wrist@414:note;"
        "state@415:note"
    )

    supporting_robot = parse_robot_refs(
        ask(
            "supporting robot refs",
            robot_refs_to_text(
                record.get(
                    "supporting_robot_refs",
                    [],
                )
            ),
        ),
        episode_id,
    )

    counter_robot = parse_robot_refs(
        ask(
            "counterevidence robot refs",
            robot_refs_to_text(
                record.get(
                    "counterevidence_robot_refs",
                    [],
                )
            ),
        ),
        episode_id,
    )

    print()
    print(
        "Manual ref syntax: "
        "4:relevance note;"
        "8:relevance note"
    )

    supporting_manual = parse_manual_refs(
        ask(
            "supporting manual refs",
            manual_refs_to_text(
                record.get(
                    "supporting_manual_refs",
                    [],
                )
            ),
        )
    )

    counter_manual = parse_manual_refs(
        ask(
            "counterevidence manual refs",
            manual_refs_to_text(
                record.get(
                    "counterevidence_manual_refs",
                    [],
                )
            ),
        )
    )

    if answerability == "insufficient_evidence":
        uncertainty = ask(
            "explicit uncertainty reason",
            record.get(
                "explicit_uncertainty_reason"
            )
            or None,
        )

    confidence_text = ask(
        "blind confidence [0,1]",
        (
            str(
                record[
                    "blind_confidence"
                ]
            )
            if record.get(
                "blind_confidence"
            ) is not None
            else None
        ),
    )

    confidence = float(
        confidence_text
    )

    notes = ask(
        "blind review notes",
        record.get(
            "blind_review_notes"
        )
        or "",
    )

    candidate = {
        **record,
        "observed_symptom": symptom,
        "failure_interval": interval,
        "supporting_robot_refs": supporting_robot,
        "counterevidence_robot_refs": counter_robot,
        "supporting_manual_refs": supporting_manual,
        "counterevidence_manual_refs": counter_manual,
        "evidence_answerability_gt": answerability,
        "explicit_uncertainty_reason": uncertainty,
        "blind_confidence": confidence,
        "blind_review_notes": (
            notes
            if notes
            else None
        ),
        "blind_cause_hypothesis": cause,
    }

    errors = record_errors(
        candidate,
        require_complete=True,
    )

    if errors:
        print()
        print("RECORD NOT SAVED")

        for error in errors:
            print(" -", error)

        raise SystemExit(1)

    records[
        position - 1
    ] = candidate

    save_records(records)

    print()
    print(
        f"PASS A CASE {position:02d}/90 SAVED: PASS"
    )


def status() -> None:
    verify_frozen_environment()

    records = load_records()

    complete = [
        row
        for row in records
        if is_complete(row)
    ]

    print("case_count =", len(records))
    print("complete =", len(complete))
    print(
        "remaining =",
        len(records) - len(complete),
    )

    if len(complete) != len(records):
        next_row = next(
            row
            for row in records
            if not is_complete(row)
        )

        print(
            "next_review_position =",
            next_row[
                "review_position"
            ],
        )
        print(
            "next_episode_id =",
            next_row[
                "episode_id"
            ],
        )


def validate(
    require_complete: bool,
) -> None:
    _, manifest = verify_frozen_environment()

    records = load_records()

    errors: list[str] = []

    if len(records) != 90:
        errors.append(
            f"record count={len(records)}"
        )

    ids = [
        row.get(
            "episode_id"
        )
        for row in records
    ]

    expected_ids = [
        case["episode_id"]
        for case in manifest["cases"]
    ]

    if ids != expected_ids:
        errors.append(
            "record order differs from frozen pack"
        )

    if len(set(ids)) != 90:
        errors.append(
            "episode ids not unique"
        )

    for index, record in enumerate(
        records,
        start=1,
    ):
        if record.get(
            "review_position"
        ) != index:
            errors.append(
                f"position {index}: "
                "review_position mismatch"
            )

        for error in record_errors(
            record,
            require_complete=require_complete,
        ):
            errors.append(
                f"position {index} "
                f"{record.get('episode_id')}: "
                f"{error}"
            )

    print(
        "===== DAY29 PASS A VALIDATION ====="
    )
    print("case_count =", len(records))
    print(
        "unique_episode_ids =",
        len(set(ids)),
    )
    print(
        "require_complete =",
        require_complete,
    )
    print("errors =", errors)

    if errors:
        raise SystemExit(1)

    print("DAY29 PASS A VALIDATION: PASS")


def freeze() -> None:
    contract, manifest = verify_frozen_environment()

    validate(
        require_complete=True,
    )

    if FREEZE_RECEIPT_PATH.exists():
        raise RuntimeError(
            "Pass A freeze receipt already exists"
        )

    records = load_records()

    answerability_counts = {
        value: sum(
            row[
                "evidence_answerability_gt"
            ] == value
            for row in records
        )
        for value in sorted(
            ANSWERABILITY
        )
    }

    blind_cause_counts = {
        cause: sum(
            row[
                "blind_cause_hypothesis"
            ] == cause
            for row in records
        )
        for cause in sorted(CAUSES)
    }

    receipt = {
        "schema_version": (
            "evidencemm_day29_pass_a_freeze_receipt_v1"
        ),
        "status": "blind_review_complete_admin_unrevealed",
        "tooling_commit": git_output(
            "rev-parse",
            "HEAD",
        ),
        "pass_a_operational_contract_sha256": (
            sha256_file(
                CONTRACT_PATH
            )
        ),
        "pass_a_records_path": (
            "data/annotations/day29_pass_a_records.jsonl"
        ),
        "pass_a_records_sha256": (
            sha256_file(
                RECORDS_PATH
            )
        ),
        "case_count": len(records),
        "review_order_sha256": (
            manifest[
                "review_order_sha256"
            ]
        ),
        "pack_tree_sha256": (
            contract[
                "frozen_dependencies"
            ][
                "pack_tree_sha256"
            ]
        ),
        "answerability_counts": (
            answerability_counts
        ),
        "blind_cause_hypothesis_counts": (
            blind_cause_counts
        ),
        "human_review_completed": True,
        "admin_reveal_started": False,
        "ground_truth_frozen": False,
        "future_split_materialized": False,
    }

    FREEZE_RECEIPT_PATH.write_text(
        json.dumps(
            receipt,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        "pass_a_records_sha256 =",
        receipt[
            "pass_a_records_sha256"
        ],
    )
    print(
        "answerability_counts =",
        answerability_counts,
    )
    print(
        "blind_cause_counts =",
        blind_cause_counts,
    )
    print(
        "DAY29 PASS A FREEZE RECEIPT: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser("preflight")
    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("review-next")

    review = sub.add_parser("review")
    review.add_argument(
        "--position",
        required=True,
        type=int,
    )

    validate_parser = sub.add_parser(
        "validate"
    )
    validate_parser.add_argument(
        "--require-complete",
        action="store_true",
    )

    sub.add_parser("freeze")

    args = parser.parse_args()

    if args.command == "preflight":
        contract, manifest = (
            verify_frozen_environment()
        )

        print(
            "manual_policy_commit =",
            contract[
                "frozen_dependencies"
            ][
                "manual_policy_commit"
            ],
        )
        print(
            "case_count =",
            manifest["case_count"],
        )
        print(
            "pack_tree_sha256 =",
            pack_tree_sha256(),
        )
        print(
            "review_order_sha256 =",
            manifest[
                "review_order_sha256"
            ],
        )
        print(
            "DAY29 PASS A PREFLIGHT: PASS"
        )
        return

    if args.command == "init":
        init_records()
        return

    if args.command == "status":
        status()
        return

    if args.command == "review-next":
        review_case(None)
        return

    if args.command == "review":
        review_case(
            args.position
        )
        return

    if args.command == "validate":
        validate(
            args.require_complete
        )
        return

    if args.command == "freeze":
        freeze()
        return


if __name__ == "__main__":
    main()
