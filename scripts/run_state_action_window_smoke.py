from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.state_action_selection import (
    ACTION_CHANGE,
    ACTION_SIGNAL,
    JOINT_ORDER,
    STATE_ACTION_FUSION,
    STATE_ACTION_TIE_BREAK,
    STATE_CHANGE,
    STATE_SIGNAL,
    TRACKING_GAP_ROLE,
    build_state_action_selection,
    load_state_action_samples,
    validate_source_semantics,
)
from evidencemm.temporal_evidence import EpisodeManifest
from evidencemm.temporal_slicing import load_temporal_slices


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--window-index", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(
        (
            ROOT
            / "configs/robot_sequence_evidence.yaml"
        ).read_text(encoding="utf-8")
    )
    state_action = config["state_action_selection"]

    episode_dir = Path(args.episode_dir)
    validate_source_semantics(
        episode_dir / "metadata.json"
    )
    samples = load_state_action_samples(
        episode_dir / "samples.csv",
        verify_tracking_error=bool(
            state_action["verify_tracking_error"]
        ),
    )

    manifest = EpisodeManifest.model_validate_json(
        (
            ROOT
            / config["manifest_root"]
            / f"{args.episode_id}.json"
        ).read_text(encoding="utf-8")
    )
    if len(samples) != manifest.frame_count:
        raise ValueError(
            "state/action sample count does not match episode manifest"
        )

    processed_dir = (
        ROOT
        / config["processed_root"]
        / args.episode_id
    )
    slices = load_temporal_slices(
        processed_dir / "temporal_slices.jsonl"
    )

    if not 0 <= args.window_index < len(slices):
        raise ValueError(
            f"window-index must be in [0, {len(slices) - 1}]"
        )

    temporal_slice = slices[args.window_index]
    selection, scores = build_state_action_selection(
        samples=samples,
        temporal_slice=temporal_slice,
    )

    top_scores = sorted(
        scores,
        key=lambda item: (
            -item.fused_state_action_score,
            item.frame_index,
        ),
    )[:5]

    payload = {
        "mode": "state_action_window_smoke",
        "gold_read": False,
        "episode_id": args.episode_id,
        "window_index": args.window_index,
        "slice_group_id": temporal_slice.slice_group_id,
        "window_start_sec": temporal_slice.start_sec,
        "window_end_sec": temporal_slice.end_sec,
        "candidate_frames": len(scores),
        "frozen_rule": {
            "joint_order": list(JOINT_ORDER),
            "state_signal": STATE_SIGNAL,
            "action_signal": ACTION_SIGNAL,
            "state_change": STATE_CHANGE,
            "action_change": ACTION_CHANGE,
            "fusion": STATE_ACTION_FUSION,
            "tie_break": STATE_ACTION_TIE_BREAK,
            "tracking_gap_role": TRACKING_GAP_ROLE,
            "normalization": state_action["normalization"],
            "joint_weighting": state_action["joint_weighting"],
            "verify_tracking_error": bool(
                state_action["verify_tracking_error"]
            ),
        },
        "selected": selection.model_dump(),
        "top5": [
            item.model_dump()
            for item in top_scores
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
