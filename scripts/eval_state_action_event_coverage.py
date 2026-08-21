from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.data_binding import sha256_file
from evidencemm.motion_eval import evaluate_motion_event_coverage
from evidencemm.motion_selection import build_motion_aware_selection
from evidencemm.state_action_eval import (
    evaluate_state_action_event_coverage,
)
from evidencemm.state_action_selection import (
    build_state_action_selection,
    load_state_action_samples,
    validate_source_semantics,
)
from evidencemm.temporal_evidence import (
    EpisodeManifest,
    load_frame_records,
)
from evidencemm.temporal_eval import (
    evaluate_event_coverage,
    load_annotations,
)
from evidencemm.temporal_slicing import load_temporal_slices


ROOT = Path(__file__).resolve().parents[1]


def mean_distance(rows: list[dict], key: str) -> float:
    return sum(float(item[key]) for item in rows) / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-dir", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )
    motion_config = config["motion_selection"]
    state_action_config = config["state_action_selection"]

    episode_dir = Path(args.episode_dir)
    metadata_path = episode_dir / "metadata.json"
    samples_csv_path = episode_dir / "samples.csv"

    validate_source_semantics(metadata_path)

    manifest = EpisodeManifest.model_validate_json(
        (
            ROOT
            / config["manifest_root"]
            / f"{args.episode_id}.json"
        ).read_text(encoding="utf-8")
    )

    if sha256_file(metadata_path) != manifest.metadata_sha256:
        raise ValueError("metadata.json SHA256 does not match manifest")
    if sha256_file(samples_csv_path) != manifest.samples_csv_sha256:
        raise ValueError("samples.csv SHA256 does not match manifest")

    processed_dir = (
        ROOT
        / config["processed_root"]
        / args.episode_id
    )
    records = load_frame_records(
        processed_dir / "frames.jsonl"
    )
    slices = load_temporal_slices(
        processed_dir / "temporal_slices.jsonl"
    )

    samples = load_state_action_samples(
        samples_csv_path,
        verify_tracking_error=bool(
            state_action_config["verify_tracking_error"]
        ),
    )
    if len(samples) != manifest.frame_count:
        raise ValueError(
            "state/action sample count does not match manifest"
        )

    annotations = load_annotations(
        ROOT
        / "data/eval"
        / (
            "day7_temporal_events_"
            f"{args.episode_id}.jsonl"
        )
    )

    state_action_selections = []
    motion_selections = []

    for temporal_slice in slices:
        state_action_selection, _state_scores = (
            build_state_action_selection(
                samples=samples,
                temporal_slice=temporal_slice,
            )
        )
        state_action_selections.append(
            state_action_selection
        )

        motion_selection, _motion_scores = (
            build_motion_aware_selection(
                episode_dir=episode_dir,
                manifest=manifest,
                records=records,
                temporal_slice=temporal_slice,
                resize_width=int(
                    motion_config["resize_width"]
                ),
                resize_height=int(
                    motion_config["resize_height"]
                ),
                verify_source_sha256=bool(
                    motion_config["verify_source_sha256"]
                ),
            )
        )
        motion_selections.append(motion_selection)

    if (
        len(state_action_selections)
        != len(motion_selections)
        or len(state_action_selections) != len(slices)
    ):
        raise ValueError(
            "all temporal methods must have one selection per slice"
        )

    for temporal_slice, state_sel, motion_sel in zip(
        slices,
        state_action_selections,
        motion_selections,
    ):
        for selection_name, selection in (
            ("state_action", state_sel),
            ("motion", motion_sel),
        ):
            if selection.slice_group_id != temporal_slice.slice_group_id:
                raise ValueError(
                    f"{selection_name} selection must reuse "
                    "frozen Day 7 slice identity"
                )
            if (
                selection.start_frame_index
                != temporal_slice.start_frame_index
                or selection.end_frame_index_exclusive
                != temporal_slice.end_frame_index_exclusive
            ):
                raise ValueError(
                    f"{selection_name} selection must reuse "
                    "frozen Day 7 frame window"
                )

    midpoint_result = evaluate_event_coverage(
        annotations=annotations,
        records=records,
        slices=slices,
    )
    motion_result = evaluate_motion_event_coverage(
        annotations=annotations,
        records=records,
        selections=motion_selections,
    )
    state_action_result = evaluate_state_action_event_coverage(
        annotations=annotations,
        records=records,
        selections=state_action_selections,
    )

    if not (
        len(slices)
        == motion_result.shared_sample_budget
        == state_action_result.shared_sample_budget
    ):
        raise ValueError(
            "all methods must use the same shared-sample budget"
        )

    midpoint_by_event = {
        item["event_id"]: item
        for item in midpoint_result["events"]
    }
    motion_by_event = {
        item["event_id"]: item
        for item in motion_result.events
    }
    state_action_by_event = {
        item["event_id"]: item
        for item in state_action_result.events
    }

    events = []
    for event_id, midpoint in midpoint_by_event.items():
        motion = motion_by_event[event_id]
        state_action = state_action_by_event[event_id]
        events.append(
            {
                "event_id": event_id,
                "event_type": midpoint["event_type"],
                "midpoint": {
                    "covered": midpoint["covered"],
                    "closest_frame": (
                        midpoint["closest_midpoint_frame"]
                    ),
                    "distance_ms": (
                        midpoint[
                            "closest_midpoint_to_event_center_ms"
                        ]
                    ),
                },
                "visual_motion": {
                    "covered": motion["covered"],
                    "closest_frame": (
                        motion["closest_selected_frame"]
                    ),
                    "distance_ms": (
                        motion[
                            "closest_selected_to_event_center_ms"
                        ]
                    ),
                },
                "state_action": {
                    "covered": state_action["covered"],
                    "closest_frame": (
                        state_action["closest_selected_frame"]
                    ),
                    "distance_ms": (
                        state_action[
                            "closest_selected_to_event_center_ms"
                        ]
                    ),
                },
                "state_action_delta_vs_midpoint_ms": (
                    state_action[
                        "closest_selected_to_event_center_ms"
                    ]
                    - midpoint[
                        "closest_midpoint_to_event_center_ms"
                    ]
                ),
                "state_action_delta_vs_visual_motion_ms": (
                    state_action[
                        "closest_selected_to_event_center_ms"
                    ]
                    - motion[
                        "closest_selected_to_event_center_ms"
                    ]
                ),
            }
        )

    midpoint_mean = mean_distance(
        midpoint_result["events"],
        "closest_midpoint_to_event_center_ms",
    )
    motion_mean = mean_distance(
        motion_result.events,
        "closest_selected_to_event_center_ms",
    )
    state_action_mean = mean_distance(
        state_action_result.events,
        "closest_selected_to_event_center_ms",
    )

    dominant_counts = {
        "state": 0,
        "action": 0,
        "tie": 0,
    }
    for row in state_action_result.selected_frames:
        dominant_counts[row["dominant_change_channel"]] += 1

    payload = {
        "mode": "day9_three_way_temporal_evidence_comparison",
        "episode_id": args.episode_id,
        "timestamp_source": config["timestamp_source"],
        "slice_duration_sec": float(config["slice_duration_sec"]),
        "comparison_protocol": {
            "same_episode": True,
            "same_temporal_windows": True,
            "same_verified_gold": True,
            "same_shared_sample_budget": True,
            "shared_sample_budget": len(slices),
            "evidence_image_budget": len(slices) * 2,
            "state_action_rule": {
                "state_signal": (
                    state_action_config["state_signal"]
                ),
                "action_signal": (
                    state_action_config["action_signal"]
                ),
                "state_change": (
                    state_action_config["state_change"]
                ),
                "action_change": (
                    state_action_config["action_change"]
                ),
                "fusion": state_action_config["fusion"],
                "tie_break": (
                    state_action_config["tie_break"]
                ),
                "normalization": (
                    state_action_config["normalization"]
                ),
                "joint_weighting": (
                    state_action_config["joint_weighting"]
                ),
                "tracking_gap_role": (
                    state_action_config["tracking_gap_role"]
                ),
            },
        },
        "uniform_midpoint": {
            "verified_events": midpoint_result["verified_events"],
            "covered_events": midpoint_result["covered_events"],
            "event_coverage": midpoint_result["event_coverage"],
            "mean_closest_distance_ms": midpoint_mean,
        },
        "visual_motion": {
            "verified_events": motion_result.verified_events,
            "covered_events": motion_result.covered_events,
            "event_coverage": motion_result.event_coverage,
            "mean_closest_distance_ms": motion_mean,
        },
        "state_action": {
            "verified_events": state_action_result.verified_events,
            "covered_events": state_action_result.covered_events,
            "event_coverage": state_action_result.event_coverage,
            "mean_closest_distance_ms": state_action_mean,
            "dominant_change_channel_counts": dominant_counts,
        },
        "delta_state_action_vs_midpoint": {
            "event_coverage": (
                state_action_result.event_coverage
                - midpoint_result["event_coverage"]
            ),
            "mean_closest_distance_ms": (
                state_action_mean - midpoint_mean
            ),
        },
        "delta_state_action_vs_visual_motion": {
            "event_coverage": (
                state_action_result.event_coverage
                - motion_result.event_coverage
            ),
            "mean_closest_distance_ms": (
                state_action_mean - motion_mean
            ),
        },
        "events": events,
        "state_action_selected_frames": (
            state_action_result.selected_frames
        ),
        "scope_note": (
            "Day 9 is a one-episode state/action temporal-evidence "
            "smoke comparison against the frozen Day 7 midpoint and "
            "Day 8 visual-motion baselines. No post-hoc selector "
            "tuning is performed."
        ),
    }

    selections_output = (
        processed_dir
        / "state_action_selections.jsonl"
    )
    selections_output.write_text(
        "\n".join(
            json.dumps(
                item.model_dump(),
                ensure_ascii=False,
            )
            for item in state_action_selections
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report = (
        ROOT
        / "reports"
        / (
            "day9_state_action_event_coverage_"
            f"{args.episode_id}.json"
        )
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "state_action_selections="
        f"{selections_output.relative_to(ROOT)}"
    )
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
