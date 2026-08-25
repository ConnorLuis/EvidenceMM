#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from evidencemm.data_binding import sha256_file


ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_KEYS = {
    "pair_group_id",
    "plan_row_id",
    "slot_role",
    "planned_physical_cause",
    "planned_cause",
    "planned_intervention_type",
    "intervention_type",
    "intervention_parameters",
    "intervention_applied",
    "selected_canonical",
    "technical_valid",
    "experimental_valid",
    "task_success",
    "intervention_verified",
    "physical_cause_gt",
    "diagnostic_decision_gt",
    "original_failure_reason",
    "observed_failure_mode",
    "reasons",
    "selection_reason",
    "selection_reasons",
    "metrics",
    "state_action",
}

FORBIDDEN_TEXT_PATTERNS = [
    re.compile(
        r"rcv2_g\d{2}",
        re.IGNORECASE,
    ),
    re.compile(
        r"target_offset_or_perception",
        re.IGNORECASE,
    ),
    re.compile(
        r"gripper_close_timing",
        re.IGNORECASE,
    ),
    re.compile(
        r"trajectory_execution_deviation",
        re.IGNORECASE,
    ),
    re.compile(
        r"clean_success",
        re.IGNORECASE,
    ),
    re.compile(
        r"target_mild",
        re.IGNORECASE,
    ),
    re.compile(
        r"gripper_late",
        re.IGNORECASE,
    ),
    re.compile(
        r"trajectory_mild",
        re.IGNORECASE,
    ),
    re.compile(
        r"uniform_anchor",
        re.IGNORECASE,
    ),
    re.compile(
        r"state_action_change",
        re.IGNORECASE,
    ),
    re.compile(
        r"tracking_gap",
        re.IGNORECASE,
    ),
    re.compile(
        r"gripper_action_change",
        re.IGNORECASE,
    ),
    re.compile(
        r"front_visual_motion",
        re.IGNORECASE,
    ),
    re.compile(
        r"wrist_visual_motion",
        re.IGNORECASE,
    ),
    re.compile(
        r"fused_state_action_score",
        re.IGNORECASE,
    ),
    re.compile(
        r"evidencemm-root-cause-v2-split-v3",
        re.IGNORECASE,
    ),
]


def walk_keys(
    value: Any,
    path: str = "$",
) -> list[str]:
    errors: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                errors.append(
                    f"{path}: forbidden key {key}"
                )

            errors.extend(
                walk_keys(
                    child,
                    f"{path}.{key}",
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(
            value
        ):
            errors.extend(
                walk_keys(
                    child,
                    f"{path}[{index}]",
                )
            )

    return errors


def deterministic_review_order(
    episode_ids: list[str],
    seed: str,
) -> list[str]:
    def key(episode_id: str) -> str:
        return hashlib.sha256(
            (
                seed
                + "\0"
                + episode_id
            ).encode("utf-8")
        ).hexdigest()

    return sorted(
        episode_ids,
        key=key,
    )


def order_sha(
    episode_ids: list[str],
) -> str:
    payload = (
        "\n".join(episode_ids) + "\n"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "day29_blind_review_pack.yaml"
        ),
    )

    args = parser.parse_args()

    config_path = (
        ROOT
        / args.config
    ).resolve()

    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    output_root = (
        ROOT
        / config["output"]["root"]
    ).resolve()

    population_path = (
        ROOT
        / config[
            "inputs"
        ][
            "population_records"
        ]
    ).resolve()

    population_sha = sha256_file(
        population_path
    )

    if (
        population_sha
        != config[
            "provenance"
        ][
            "frozen_blank_records_sha256"
        ]
    ):
        raise SystemExit(
            "frozen population SHA256 mismatch"
        )

    population_rows = [
        json.loads(line)
        for line in population_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    population_ids = [
        row["episode_id"]
        for row in population_rows
    ]

    source_manifest_path = (
        ROOT
        / config[
            "inputs"
        ][
            "source_manifest"
        ]
    ).resolve()

    with source_manifest_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        source_rows = list(
            csv.DictReader(handle)
        )

    source_by_episode = {}

    for row in source_rows:
        episode_id = row.get(
            "episode_id",
            "",
        )

        if not episode_id:
            raise SystemExit(
                "source manifest row lacks episode_id"
            )

        if episode_id in source_by_episode:
            raise SystemExit(
                "duplicate source manifest episode: "
                + episode_id
            )

        source_by_episode[
            episode_id
        ] = row

    raw_audit_path = (
        ROOT
        / config[
            "inputs"
        ][
            "raw_audit_config"
        ]
    ).resolve()

    raw_audit = yaml.safe_load(
        raw_audit_path.read_text(
            encoding="utf-8"
        )
    )

    raw_root = Path(
        raw_audit[
            "raw_source"
        ][
            "compatibility_wsl_root"
        ]
    ).resolve()

    errors: list[str] = []

    manifest_path = (
        output_root
        / "manifest.json"
    )

    if not manifest_path.is_file():
        raise SystemExit(
            "manifest.json missing"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    errors.extend(
        walk_keys(manifest)
    )

    expected_count = int(
        config[
            "expected"
        ][
            "canonical_episode_count"
        ]
    )

    cases = manifest.get(
        "cases",
        [],
    )

    if len(cases) != expected_count:
        errors.append(
            f"case count: {len(cases)}"
        )

    episode_ids = [
        case.get("episode_id")
        for case in cases
    ]

    if len(set(episode_ids)) != len(
        episode_ids
    ):
        errors.append(
            "duplicate episode IDs"
        )

    expected_order = (
        deterministic_review_order(
            population_ids,
            config[
                "review_order"
            ][
                "seed"
            ],
        )
    )

    if episode_ids != expected_order:
        errors.append(
            "review order differs from frozen "
            "population + seed"
        )

    if (
        order_sha(episode_ids)
        != manifest.get(
            "review_order_sha256"
        )
    ):
        errors.append(
            "review order SHA mismatch"
        )

    if (
        manifest.get(
            "review_order_method"
        )
        != config[
            "review_order"
        ][
            "method"
        ]
    ):
        errors.append(
            "review order method mismatch"
        )

    if (
        manifest.get(
            "review_order_seed"
        )
        != config[
            "review_order"
        ][
            "seed"
        ]
    ):
        errors.append(
            "review order seed mismatch"
        )

    if manifest.get(
        "human_review_started"
    ) is not False:
        errors.append(
            "human_review_started is not false"
        )

    if manifest.get(
        "ground_truth_frozen"
    ) is not False:
        errors.append(
            "ground_truth_frozen is not false"
        )

    if manifest.get(
        "future_split_materialized"
    ) is not False:
        errors.append(
            "future split unexpectedly materialized"
        )

    selected_total = 0

    expected_frames = int(
        config[
            "expected"
        ][
            "frame_count_per_episode"
        ]
    )

    for case in cases:
        episode_id = case[
            "episode_id"
        ]

        case_dir = (
            output_root
            / episode_id
        )

        source_row = source_by_episode.get(
            episode_id
        )

        expected_episode_dir = None

        if source_row is None:
            errors.append(
                f"{episode_id}: "
                "missing Day28 source binding"
            )
        else:
            expected_episode_dir = (
                raw_root
                / source_row[
                    "raw_episode_relpath"
                ]
            ).resolve()

        required = [
            "review.html",
            "review_context.json",
            "selected_frames.json",
            "selected_frames.csv",
            "frame_explorer.html",
            "full_state_action.json",
        ]

        for name in required:
            if not (
                case_dir / name
            ).is_file():
                errors.append(
                    f"{episode_id}: "
                    f"missing {name}"
                )

        selected_path = (
            case_dir
            / "selected_frames.json"
        )

        if not selected_path.is_file():
            continue

        actual_selected_sha = sha256_file(
            selected_path
        )

        if (
            case.get(
                "selected_frames_sha256"
            )
            != actual_selected_sha
        ):
            errors.append(
                f"{episode_id}: "
                "selected_frames SHA256 mismatch"
            )

        payload = json.loads(
            selected_path.read_text(
                encoding="utf-8"
            )
        )

        errors.extend(
            walk_keys(
                payload,
                episode_id,
            )
        )

        rows = payload.get(
            "rows",
            [],
        )

        count = len(rows)

        selected_total += count

        if (
            case.get(
                "full_frame_count"
            )
            != expected_frames
        ):
            errors.append(
                f"{episode_id}: "
                "manifest full_frame_count mismatch"
            )

        if not (
            int(
                config[
                    "selection"
                ][
                    "uniform_count"
                ]
            )
            <= count
            <= int(
                config[
                    "selection"
                ][
                    "max_selected_frames"
                ]
            )
        ):
            errors.append(
                f"{episode_id}: "
                f"selected count {count}"
            )

        frame_ids = [
            row.get(
                "frame_index"
            )
            for row in rows
        ]

        if len(set(frame_ids)) != len(
            frame_ids
        ):
            errors.append(
                f"{episode_id}: "
                "duplicate selected frames"
            )

        thumbs = (
            case_dir
            / "thumbs"
        )

        thumb_count = len(
            list(
                thumbs.glob("*.jpg")
            )
        )

        if thumb_count != (
            2 * count
        ):
            errors.append(
                f"{episode_id}: "
                f"thumbnail count "
                f"{thumb_count}"
            )

        context_path = (
            case_dir
            / "review_context.json"
        )

        if context_path.is_file():
            context = json.loads(
                context_path.read_text(
                    encoding="utf-8"
                )
            )

            errors.extend(
                walk_keys(
                    context,
                    (
                        episode_id
                        + ".context"
                    ),
                )
            )


        full_state_path = (
            case_dir
            / "full_state_action.json"
        )

        if full_state_path.is_file():
            actual_full_state_sha = (
                sha256_file(
                    full_state_path
                )
            )

            if (
                case.get(
                    "full_state_action_sha256"
                )
                != actual_full_state_sha
            ):
                errors.append(
                    f"{episode_id}: "
                    "full_state_action SHA256 mismatch"
                )

            full_state = json.loads(
                full_state_path.read_text(
                    encoding="utf-8"
                )
            )

            errors.extend(
                walk_keys(
                    full_state,
                    (
                        episode_id
                        + ".full_state"
                    ),
                )
            )

            full_rows = full_state.get(
                "rows",
                [],
            )

            expected_frames = int(
                config[
                    "expected"
                ][
                    "frame_count_per_episode"
                ]
            )

            if (
                full_state.get("frame_count")
                != expected_frames
            ):
                errors.append(
                    f"{episode_id}: "
                    "full-state frame_count mismatch"
                )

            if len(full_rows) != expected_frames:
                errors.append(
                    f"{episode_id}: "
                    f"full-state rows={len(full_rows)}"
                )

            full_ids = [
                row.get("frame_index")
                for row in full_rows
            ]

            if full_ids != list(
                range(expected_frames)
            ):
                errors.append(
                    f"{episode_id}: "
                    "full-state frame indices "
                    "are not exactly 0..899"
                )

            allowed_row_keys = {
                "frame_index",
                "timestamp_sec",
                "observation",
                "action",
                "tracking_error",
            }

            for index, row in enumerate(
                full_rows
            ):
                if set(row) != allowed_row_keys:
                    errors.append(
                        f"{episode_id}: "
                        f"full-state row {index} "
                        f"keys={sorted(row)}"
                    )
                    break

        for camera, expected_name in (
            ("front_frames", "front"),
            ("wrist_frames", "wrist"),
        ):
            link = case_dir / camera

            if not link.is_symlink():
                errors.append(
                    f"{episode_id}: "
                    f"{camera} is not symlink"
                )
                continue

            target = link.resolve()

            if expected_episode_dir is not None:
                expected_target = (
                    expected_episode_dir
                    / expected_name
                ).resolve()

                if target != expected_target:
                    errors.append(
                        f"{episode_id}: "
                        f"{camera} target mismatch"
                    )

            jpg_count = len(
                list(
                    link.glob("*.jpg")
                )
            )

            if jpg_count != int(
                config[
                    "expected"
                ][
                    (
                        "front_images_per_episode"
                        if camera == "front_frames"
                        else "wrist_images_per_episode"
                    )
                ]
            ):
                errors.append(
                    f"{episode_id}: "
                    f"{camera} jpg_count="
                    f"{jpg_count}"
                )

    manual_manifest = (
        output_root
        / "manual"
        / "manifest.json"
    )

    if not manual_manifest.is_file():
        errors.append(
            "manual manifest missing"
        )
    else:
        manual = json.loads(
            manual_manifest.read_text(
                encoding="utf-8"
            )
        )

        errors.extend(
            walk_keys(
                manual,
                "$.manual",
            )
        )

        if len(
            manual.get(
                "pages",
                [],
            )
        ) != int(
            manual.get(
                "page_count",
                -1,
            )
        ):
            errors.append(
                "manual page count mismatch"
            )

    text_files = [
        path
        for path in output_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in {
                ".json",
                ".csv",
                ".html",
            }
        )
    ]

    for path in text_files:
        text = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        for pattern in (
            FORBIDDEN_TEXT_PATTERNS
        ):
            if pattern.search(text):
                errors.append(
                    f"{path.relative_to(output_root)}: "
                    f"forbidden text "
                    f"{pattern.pattern}"
                )

    print(
        "===== DAY29 BLIND PACK VALIDATION ====="
    )
    print(
        "case_count =",
        len(cases),
    )
    print(
        "unique_episode_ids =",
        len(set(episode_ids)),
    )
    print(
        "selected_frame_total =",
        selected_total,
    )
    print(
        "review_order_sha_match =",
        (
            order_sha(episode_ids)
            == manifest.get(
                "review_order_sha256"
            )
        ),
    )
    print(
        "errors =",
        errors,
    )

    if errors:
        return 1

    print(
        "BLIND REVIEW PACK VALIDATION: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
