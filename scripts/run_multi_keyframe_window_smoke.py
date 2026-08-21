from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.multi_keyframe_selection import (
    DUPLICATE_POLICY,
    FROZEN_K_VALUES,
    NEAREST_SAMPLE_TIE_BREAK,
    SELECTION_SIGNALS,
    TARGET_RULE,
    build_multi_keyframe_selection,
)
from evidencemm.temporal_evidence import (
    load_frame_records,
)
from evidencemm.temporal_slicing import (
    load_temporal_slices,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-id",
        required=True,
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )
    multi = config["multi_keyframe_selection"]

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

    if not 0 <= args.window_index < len(slices):
        raise ValueError(
            f"window-index must be in [0, {len(slices) - 1}]"
        )

    temporal_slice = slices[args.window_index]

    selections = [
        build_multi_keyframe_selection(
            records=records,
            temporal_slice=temporal_slice,
            k=k,
        )
        for k in FROZEN_K_VALUES
    ]

    k1 = selections[0]
    if (
        k1.keyframes[0].selected_frame_index
        != temporal_slice.midpoint_frame_index
    ):
        raise ValueError(
            "K=1 must reproduce the frozen Day 7 midpoint frame"
        )
    if (
        abs(
            k1.keyframes[0].selected_timestamp_sec
            - temporal_slice.midpoint_timestamp_sec
        )
        > 1e-12
    ):
        raise ValueError(
            "K=1 must reproduce the frozen Day 7 midpoint timestamp"
        )

    payload = {
        "mode": "multi_keyframe_window_smoke",
        "gold_read": False,
        "episode_id": args.episode_id,
        "window_index": args.window_index,
        "slice_group_id": temporal_slice.slice_group_id,
        "window_start_sec": temporal_slice.start_sec,
        "window_end_sec": temporal_slice.end_sec,
        "candidate_frames": (
            temporal_slice.end_frame_index_exclusive
            - temporal_slice.start_frame_index
        ),
        "frozen_protocol": {
            "k_values": list(FROZEN_K_VALUES),
            "target_rule": TARGET_RULE,
            "nearest_sample_tie_break": (
                NEAREST_SAMPLE_TIE_BREAK
            ),
            "selection_signals": SELECTION_SIGNALS,
            "duplicate_policy": DUPLICATE_POLICY,
            "camera_pairing": multi["camera_pairing"],
            "report_all_k": bool(multi["report_all_k"]),
            "choose_best_k": bool(multi["choose_best_k"]),
        },
        "day7_midpoint_reference": {
            "frame_index": (
                temporal_slice.midpoint_frame_index
            ),
            "timestamp_sec": (
                temporal_slice.midpoint_timestamp_sec
            ),
        },
        "selections": [
            item.model_dump()
            for item in selections
        ],
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
