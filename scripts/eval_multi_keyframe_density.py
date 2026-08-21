from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.multi_keyframe_eval import (
    evaluate_multi_keyframe_event_coverage,
)
from evidencemm.multi_keyframe_selection import (
    FROZEN_K_VALUES,
    build_multi_keyframe_selection,
)
from evidencemm.temporal_evidence import (
    load_frame_records,
)
from evidencemm.temporal_eval import (
    evaluate_event_coverage,
    load_annotations,
)
from evidencemm.temporal_slicing import (
    load_temporal_slices,
)


ROOT = Path(__file__).resolve().parents[1]


def mean_distance_ms(events: list[dict]) -> float:
    return (
        sum(
            float(
                item[
                    "closest_selected_to_event_center_ms"
                ]
            )
            for item in events
        )
        / len(events)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-id",
        required=True,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )
    multi = config["multi_keyframe_selection"]

    configured_k = tuple(
        int(value)
        for value in multi["k_values"]
    )
    if configured_k != FROZEN_K_VALUES:
        raise ValueError(
            "configured K values differ from frozen Day 10 protocol"
        )
    if not bool(multi["report_all_k"]):
        raise ValueError(
            "Day 10 requires report_all_k=true"
        )
    if bool(multi["choose_best_k"]):
        raise ValueError(
            "Day 10 requires choose_best_k=false"
        )
    if multi["selection_signals"] != "timestamp_only":
        raise ValueError(
            "Day 10 selection must remain timestamp-only"
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

    day7 = evaluate_event_coverage(
        annotations=annotations,
        records=records,
        slices=slices,
    )

    selections_by_k = {}
    results_by_k = {}

    for k in FROZEN_K_VALUES:
        selections = [
            build_multi_keyframe_selection(
                records=records,
                temporal_slice=temporal_slice,
                k=k,
            )
            for temporal_slice in slices
        ]

        if len(selections) != len(slices):
            raise ValueError(
                "selection count must match frozen slice count"
            )

        for temporal_slice, selection in zip(
            slices,
            selections,
        ):
            if selection.slice_group_id != temporal_slice.slice_group_id:
                raise ValueError(
                    "selection must reuse frozen slice identity"
                )
            if (
                selection.start_frame_index
                != temporal_slice.start_frame_index
                or selection.end_frame_index_exclusive
                != temporal_slice.end_frame_index_exclusive
            ):
                raise ValueError(
                    "selection must reuse frozen frame window"
                )

        if k == 1:
            for temporal_slice, selection in zip(
                slices,
                selections,
            ):
                keyframe = selection.keyframes[0]
                if (
                    keyframe.selected_frame_index
                    != temporal_slice.midpoint_frame_index
                ):
                    raise ValueError(
                        "K=1 frame must reproduce Day 7 midpoint"
                    )
                if (
                    abs(
                        keyframe.selected_timestamp_sec
                        - temporal_slice.midpoint_timestamp_sec
                    )
                    > 1e-12
                ):
                    raise ValueError(
                        "K=1 timestamp must reproduce Day 7 midpoint"
                    )

        result = evaluate_multi_keyframe_event_coverage(
            annotations=annotations,
            records=records,
            selections=selections,
        )

        selections_by_k[k] = selections
        results_by_k[k] = result

    k1 = results_by_k[1]
    if (
        k1.verified_events != day7["verified_events"]
        or k1.covered_events != day7["covered_events"]
        or abs(k1.event_coverage - day7["event_coverage"]) > 1e-12
    ):
        raise ValueError(
            "K=1 coverage must reproduce frozen Day 7 baseline"
        )

    day7_events = {
        item["event_id"]: item
        for item in day7["events"]
    }
    k1_events = {
        item["event_id"]: item
        for item in k1.events
    }
    for event_id, baseline in day7_events.items():
        current = k1_events[event_id]
        if (
            baseline["closest_midpoint_frame"]
            != current["closest_selected_frame"]
        ):
            raise ValueError(
                "K=1 closest frame must reproduce Day 7"
            )
        if (
            abs(
                baseline[
                    "closest_midpoint_to_event_center_ms"
                ]
                - current[
                    "closest_selected_to_event_center_ms"
                ]
            )
            > 1e-9
        ):
            raise ValueError(
                "K=1 temporal distance must reproduce Day 7"
            )

    summaries = {}
    for k in FROZEN_K_VALUES:
        result = results_by_k[k]
        summaries[str(k)] = {
            "k": k,
            "verified_events": result.verified_events,
            "covered_events": result.covered_events,
            "event_coverage": result.event_coverage,
            "mean_closest_distance_ms": (
                mean_distance_ms(result.events)
            ),
            "window_count": result.window_count,
            "shared_sample_budget": (
                result.shared_sample_budget
            ),
            "evidence_image_budget": (
                result.evidence_image_budget
            ),
        }

    event_ids = [
        item["event_id"]
        for item in day7["events"]
    ]
    event_rows = []

    for event_id in event_ids:
        row = {
            "event_id": event_id,
            "event_type": day7_events[event_id]["event_type"],
        }
        for k in FROZEN_K_VALUES:
            event = {
                item["event_id"]: item
                for item in results_by_k[k].events
            }[event_id]
            row[f"k{k}"] = {
                "covered": event["covered"],
                "closest_frame": (
                    event["closest_selected_frame"]
                ),
                "closest_keyframe_rank": (
                    event[
                        "closest_selected_keyframe_rank"
                    ]
                ),
                "distance_ms": (
                    event[
                        "closest_selected_to_event_center_ms"
                    ]
                ),
                "covering_frames": (
                    event["covering_selected_frames"]
                ),
            }
        event_rows.append(row)

    marginal = []
    for previous_k, current_k in ((1, 2), (2, 3)):
        previous = summaries[str(previous_k)]
        current = summaries[str(current_k)]
        marginal.append(
            {
                "from_k": previous_k,
                "to_k": current_k,
                "additional_shared_samples": (
                    current["shared_sample_budget"]
                    - previous["shared_sample_budget"]
                ),
                "additional_evidence_images": (
                    current["evidence_image_budget"]
                    - previous["evidence_image_budget"]
                ),
                "event_coverage_delta": (
                    current["event_coverage"]
                    - previous["event_coverage"]
                ),
                "covered_events_delta": (
                    current["covered_events"]
                    - previous["covered_events"]
                ),
                "mean_closest_distance_delta_ms": (
                    current["mean_closest_distance_ms"]
                    - previous["mean_closest_distance_ms"]
                ),
            }
        )

    payload = {
        "mode": "day10_multi_keyframe_density_diagnostic",
        "episode_id": args.episode_id,
        "timestamp_source": config["timestamp_source"],
        "slice_duration_sec": float(
            config["slice_duration_sec"]
        ),
        "diagnostic_status": (
            "known_frozen_gold_not_blind_benchmark"
        ),
        "frozen_protocol": {
            "k_values": list(FROZEN_K_VALUES),
            "target_rule": multi["target_rule"],
            "nearest_sample_tie_break": (
                multi["nearest_sample_tie_break"]
            ),
            "selection_signals": (
                multi["selection_signals"]
            ),
            "duplicate_policy": (
                multi["duplicate_policy"]
            ),
            "report_all_k": bool(
                multi["report_all_k"]
            ),
            "choose_best_k": bool(
                multi["choose_best_k"]
            ),
        },
        "k1_reproduces_day7_midpoint": True,
        "results_by_k": summaries,
        "marginal_gain": marginal,
        "events": event_rows,
        "selected_frames_by_k": {
            str(k): results_by_k[k].selected_frames
            for k in FROZEN_K_VALUES
        },
        "scope_note": (
            "Day 10 is the final temporal micro-baseline. "
            "All K values are reported; no production K is chosen "
            "from this three-event diagnostic and no post-hoc "
            "quantile or gold tuning is performed."
        ),
    }

    for k in FROZEN_K_VALUES:
        output = (
            processed_dir
            / f"multi_keyframe_selections_k{k}.jsonl"
        )
        output.write_text(
            "\n".join(
                json.dumps(
                    item.model_dump(),
                    ensure_ascii=False,
                )
                for item in selections_by_k[k]
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    report = (
        ROOT
        / "reports"
        / (
            "day10_multi_keyframe_density_"
            f"{args.episode_id}.json"
        )
    )
    report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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
    for k in FROZEN_K_VALUES:
        print(
            f"k{k}_selections="
            f"data/processed/robot_sequence/"
            f"{args.episode_id}/"
            f"multi_keyframe_selections_k{k}.jsonl"
        )
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
