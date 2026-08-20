from __future__ import annotations

import argparse
from pathlib import Path

from evidencemm.temporal_eval import (
    TemporalEventAnnotation,
    save_annotations,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-id",
        required=True,
    )
    args = parser.parse_args()

    output = (
        ROOT
        / "data/eval"
        / (
            "day7_temporal_events_"
            f"{args.episode_id}.jsonl"
        )
    )

    if output.exists():
        raise SystemExit(
            f"annotation file already exists: {output}"
        )

    annotations = [
        TemporalEventAnnotation(
            event_id="d7_evt_lift",
            episode_id=args.episode_id,
            event_type="object_lift",
            description=(
                "红色方块从桌面开始离开，"
                "直到形成清晰稳定的离桌状态"
            ),
            status="draft",
            start_frame_index=None,
            end_frame_index_inclusive=None,
            evidence_cameras=[
                "front",
                "wrist",
            ],
            notes=(
                "人工查看原始 front/wrist 图片后填写高置信度帧范围"
            ),
        ),
        TemporalEventAnnotation(
            event_id="d7_evt_transport",
            episode_id=args.episode_id,
            event_type="object_transport",
            description=(
                "方块保持稳定离桌并随夹爪持续移动，"
                "直到进入目标区域上方的对准与下降过渡阶段"
            ),
            status="draft",
            start_frame_index=None,
            end_frame_index_inclusive=None,
            evidence_cameras=[
                "front",
                "wrist",
            ],
            notes=(
                "人工查看原始 front/wrist 图片后填写高置信度帧范围"
            ),
        ),
        TemporalEventAnnotation(
            event_id="d7_evt_place",
            episode_id=args.episode_id,
            event_type="object_place",
            description=(
                "方块已经由目标表面明确承托，"
                "并在目标区域形成稳定放置状态"
            ),
            status="draft",
            start_frame_index=None,
            end_frame_index_inclusive=None,
            evidence_cameras=[
                "front",
                "wrist",
            ],
            notes=(
                "人工查看原始 front/wrist 图片后填写高置信度帧范围；"
                "若 release 完成边界不可可靠观察，不强行标注 release"
            ),
        ),
    ]

    save_annotations(
        annotations,
        output,
    )

    print(output.relative_to(ROOT))
    print(
        "Created 3 draft temporal events: "
        "object_lift, object_transport, object_place. "
        "Inspect original paired images, fill high-confidence "
        "frame ranges, then set status=verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
