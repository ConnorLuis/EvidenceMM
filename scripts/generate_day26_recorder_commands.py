#!/usr/bin/env python3
from pathlib import Path
from evidencemm.day24_target_collection import load_csv

RECORDER = r"""python scripts\windows\record_episode_150_windows_v7.py `
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
    --task "抓取无压纹红色方块并放入固定目标区" """

def main():
    rows=load_csv(Path("data/protocol/day26_trajectory_collection_plan.csv"))
    for i,row in enumerate(rows,1):
        label=" repeat" if row["repeat_slot"]=="true" else ""
        print("="*72)
        print(f"[{i:02d}/20] {row['plan_row_id']}{label}")
        print("方块保持 nominal；夹爪保持 clean 的正常闭合时序。")
        print("正常接近至方块正上方附近后，仅沿 Follower 正前方将末端路径偏移约40–60mm。")
        print("保持该偏移继续下降；不移动方块、不提前/延迟闭合、不增加第二偏轨、不补偿回来。")
        print("要求：任务失败。")
        print(RECORDER)
    return 0
if __name__=="__main__":
    raise SystemExit(main())
