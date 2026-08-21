from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.motion_eval import (
    evaluate_motion_event_coverage,
)
from evidencemm.motion_selection import (
    build_motion_aware_selection,
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
    motion = config["motion_selection"]

    manifest = EpisodeManifest.model_validate_json(
        (
            ROOT
            / config["manifest_root"]
            / f"{args.episode_id}.json"
        ).read_text(encoding="utf-8")
    )

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

    annotations = load_annotations(
        ROOT
        / "data/eval"
        / (
            "day7_temporal_events_"
            f"{args.episode_id}.jsonl"
        )
    )

    selections = []
    for temporal_slice in slices:
        selection, _scores = build_motion_aware_selection(
            episode_dir=Path(args.episode_dir),
            manifest=manifest,
            records=records,
            temporal_slice=temporal_slice,
            resize_width=int(motion["resize_width"]),
            resize_height=int(motion["resize_height"]),
            verify_source_sha256=bool(
                motion["verify_source_sha256"]
            ),
        )
        selections.append(selection)

    if len(selections) != len(slices):
        raise ValueError(
            "motion selection count must match frozen temporal slice count"
        )

    for temporal_slice, selection in zip(slices, selections):
        if selection.slice_group_id != temporal_slice.slice_group_id:
            raise ValueError(
                "motion selection must reuse frozen Day 7 slice identity"
            )
        if (
            selection.start_frame_index
            != temporal_slice.start_frame_index
            or selection.end_frame_index_exclusive
            != temporal_slice.end_frame_index_exclusive
        ):
            raise ValueError(
                "motion selection must reuse frozen Day 7 frame window"
            )

    motion_result = evaluate_motion_event_coverage(
        annotations=annotations,
        records=records,
        selections=selections,
    )

    midpoint_result = evaluate_event_coverage(
        annotations=annotations,
        records=records,
        slices=slices,
    )

    if motion_result.shared_sample_budget != len(slices):
        raise ValueError(
            "motion budget must equal one shared sample per Day 7 slice"
        )

    midpoint_events = {
        item["event_id"]: item
        for item in midpoint_result["events"]
    }
    motion_events = {
        item["event_id"]: item
        for item in motion_result.events
    }

    per_event_comparison = []
    for event_id in midpoint_events:
        baseline = midpoint_events[event_id]
        current = motion_events[event_id]
        per_event_comparison.append(
            {
                "event_id": event_id,
                "event_type": baseline["event_type"],
                "midpoint_covered": baseline["covered"],
                "motion_covered": current["covered"],
                "midpoint_closest_frame": (
                    baseline["closest_midpoint_frame"]
                ),
                "motion_closest_frame": (
                    current["closest_selected_frame"]
                ),
                "midpoint_distance_ms": (
                    baseline[
                        "closest_midpoint_to_event_center_ms"
                    ]
                ),
                "motion_distance_ms": (
                    current[
                        "closest_selected_to_event_center_ms"
                    ]
                ),
                "distance_delta_ms": (
                    current[
                        "closest_selected_to_event_center_ms"
                    ]
                    - baseline[
                        "closest_midpoint_to_event_center_ms"
                    ]
                ),
            }
        )

    midpoint_mean_distance_ms = (
        sum(
            item["closest_midpoint_to_event_center_ms"]
            for item in midpoint_result["events"]
        )
        / midpoint_result["verified_events"]
    )
    motion_mean_distance_ms = (
        sum(
            item["closest_selected_to_event_center_ms"]
            for item in motion_result.events
        )
        / motion_result.verified_events
    )

    payload = {
        "mode": "day8_visual_motion_vs_uniform_midpoint",
        "episode_id": args.episode_id,
        "timestamp_source": config["timestamp_source"],
        "slice_duration_sec": float(config["slice_duration_sec"]),
        "comparison_protocol": {
            "same_episode": True,
            "same_temporal_windows": True,
            "same_verified_gold": True,
            "same_shared_sample_budget": True,
            "shared_sample_budget": motion_result.shared_sample_budget,
            "evidence_image_budget": motion_result.evidence_image_budget,
            "motion_rule": {
                "resize": [
                    int(motion["resize_width"]),
                    int(motion["resize_height"]),
                ],
                "grayscale": bool(motion["grayscale"]),
                "difference": motion["difference"],
                "fusion": motion["fusion"],
                "tie_break": motion["tie_break"],
                "verify_source_sha256": bool(
                    motion["verify_source_sha256"]
                ),
            },
        },
        "uniform_midpoint": {
            "verified_events": midpoint_result["verified_events"],
            "covered_events": midpoint_result["covered_events"],
            "event_coverage": midpoint_result["event_coverage"],
            "mean_closest_distance_ms": midpoint_mean_distance_ms,
        },
        "visual_motion": {
            "verified_events": motion_result.verified_events,
            "covered_events": motion_result.covered_events,
            "event_coverage": motion_result.event_coverage,
            "mean_closest_distance_ms": motion_mean_distance_ms,
        },
        "delta": {
            "event_coverage": (
                motion_result.event_coverage
                - midpoint_result["event_coverage"]
            ),
            "mean_closest_distance_ms": (
                motion_mean_distance_ms
                - midpoint_mean_distance_ms
            ),
        },
        "events": per_event_comparison,
        "motion_selected_frames": motion_result.selected_frames,
        "scope_note": (
            "Day 8 is a one-episode visual-motion smoke comparison "
            "against the frozen Day 7 uniform-midpoint baseline. "
            "No q_t/action signal is used."
        ),
    }

    selection_output = (
        processed_dir
        / "motion_selections.jsonl"
    )
    selection_output.write_text(
        "\n".join(
            json.dumps(
                item.model_dump(),
                ensure_ascii=False,
            )
            for item in selections
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report = (
        ROOT
        / "reports"
        / (
            "day8_motion_event_coverage_"
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
        "motion_selections="
        f"{selection_output.relative_to(ROOT)}"
    )
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
