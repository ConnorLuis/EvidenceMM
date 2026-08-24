#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from evidencemm.day24_target_collection import load_csv

COMMAND = r'''python scripts\windows\record_episode_150_windows_v7.py `
    --front-index 0 `
    --wrist-index 1 `
    --duration 60 `
    --hz 15 `
    --countdown 5 `
    --motion-speed-scale 1.25 `
    --gripper-speed-scale 1.0 `
    --pose configs\poses\follower_home_v2.json `
    --leader-pose configs\poses\leader_episode_home_v1.json `
    --output-dir outputs\episodes_root_cause_v2_final `
    --task "抓取无压纹红色方块并放入固定目标区"'''

INSTRUCTIONS = {
    "target_mild_20mm_forward": (
        "方块从 nominal marker 沿 Follower 正前方移动 20 mm；机械臂仍执行原 nominal 抓取位置；不追踪、不补偿。"
    ),
    "gripper_late_30_40mm_upward_progress": (
        "方块与空间路径保持 nominal；经过正常闭合点后继续上移约 30-40 mm，再闭合夹爪；不得加入轨迹偏移。"
    ),
    "trajectory_mild_25mm_forward": (
        "方块保持 nominal；正常接近至方块上方后，末端沿 Follower 正前方偏移约 25 mm 并保持偏移下降；夹爪正常时序；不补偿回抓。"
    ),
}

def main():
    plan=load_csv(Path("data/protocol/day27_insufficient_evidence_collection_plan.csv"))
    for row in plan:
        variant=row["ambiguity_protocol"].split(":",1)[1]
        print("="*72)
        print(f"{row['day27_sequence']}. {row['plan_row_id']}  variant={variant}")
        print("ADMIN-ONLY OPERATOR RULE:", INSTRUCTIONS[variant])
        print("Only this ONE predeclared challenge is allowed. No second intervention. No evidence corruption.")
        print("Day27 only collects the candidate; answerability is judged blind on Day29.")
        print(COMMAND)
    return 0
if __name__=="__main__": raise SystemExit(main())
