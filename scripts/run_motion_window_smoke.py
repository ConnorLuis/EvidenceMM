from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evidencemm.motion_selection import (
    MOTION_DIFFERENCE,
    MOTION_FUSION,
    MOTION_RESIZE_HEIGHT,
    MOTION_RESIZE_WIDTH,
    MOTION_TIE_BREAK,
    build_motion_aware_selection,
)
from evidencemm.temporal_evidence import (
    EpisodeManifest,
    load_frame_records,
)
from evidencemm.temporal_slicing import load_temporal_slices


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--window-index", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(
        (ROOT / "configs/robot_sequence_evidence.yaml").read_text(
            encoding="utf-8"
        )
    )
    motion = config["motion_selection"]
    manifest = EpisodeManifest.model_validate_json(
        (
            ROOT
            / config["manifest_root"]
            / f"{args.episode_id}.json"
        ).read_text(encoding="utf-8")
    )
    processed_dir = ROOT / config["processed_root"] / args.episode_id
    records = load_frame_records(processed_dir / "frames.jsonl")
    slices = load_temporal_slices(processed_dir / "temporal_slices.jsonl")

    if not 0 <= args.window_index < len(slices):
        raise ValueError(
            f"window-index must be in [0, {len(slices) - 1}]"
        )

    temporal_slice = slices[args.window_index]
    selection, scores = build_motion_aware_selection(
        episode_dir=Path(args.episode_dir),
        manifest=manifest,
        records=records,
        temporal_slice=temporal_slice,
        resize_width=int(motion["resize_width"]),
        resize_height=int(motion["resize_height"]),
        verify_source_sha256=bool(motion["verify_source_sha256"]),
    )
    top_scores = sorted(
        scores,
        key=lambda item: (-item.fused_motion_score, item.frame_index),
    )[:5]

    print(
        json.dumps(
            {
                "mode": "visual_motion_window_smoke",
                "gold_read": False,
                "episode_id": args.episode_id,
                "window_index": args.window_index,
                "slice_group_id": temporal_slice.slice_group_id,
                "window_start_sec": temporal_slice.start_sec,
                "window_end_sec": temporal_slice.end_sec,
                "candidate_frames": len(scores),
                "preprocessing": {
                    "orientation": "manifest_camera_transform",
                    "grayscale": True,
                    "resize_width": int(motion["resize_width"]),
                    "resize_height": int(motion["resize_height"]),
                    "difference": MOTION_DIFFERENCE,
                    "fusion": MOTION_FUSION,
                    "tie_break": MOTION_TIE_BREAK,
                    "verify_source_sha256": bool(
                        motion["verify_source_sha256"]
                    ),
                },
                "frozen_defaults": {
                    "resize_width": MOTION_RESIZE_WIDTH,
                    "resize_height": MOTION_RESIZE_HEIGHT,
                },
                "selected": selection.model_dump(),
                "top5": [item.model_dump() for item in top_scores],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
