#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from evidencemm.day24_target_collection import load_csv


PLAN = Path("data/protocol/day24_target_collection_plan.csv")

RECORDER = r"""python scripts\windows\record_episode_150_windows_v7.py `
    --duration 60 `
    --hz 15 `
    --countdown 5 `
    --motion-speed-scale 1.25 `
    --gripper-speed-scale 1.0 `
    --pose configs\poses\follower_home_v2.json `
    --leader-pose configs\poses\leader_episode_home_v1.json `
    --output-dir outputs\episodes_root_cause_v2_final `
    --task "抓取无压纹红色方块并放入固定目标区"
"""


def main() -> int:
    rows = load_csv(PLAN)
    for row in rows:
        print("=" * 72)
        print(
            f"{row['day24_sequence']:>2}/35  "
            f"{row['plan_row_id']}  pair={row['pair_group_id']}"
        )
        if row["slot_role"] == "clean_control":
            print("操作：方块放在 nominal 标记框；正常抓取、正常放置；不做任何干预。")
            print("期望：任务成功。")
        else:
            print(
                "操作：方块从 nominal 标记沿 Follower 正前方移动 40 mm；"
                "机械臂仍抓原 nominal 位置；不追踪、不补偿。"
            )
            print("期望：任务失败。")
        print()
        print(RECORDER)
        print(
            "录完后：记录 Episode ID，并用 update_day24_target_record.py 登记。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
