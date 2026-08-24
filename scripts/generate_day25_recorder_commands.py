#!/usr/bin/env python3
from pathlib import Path
from evidencemm.day24_target_collection import load_csv

RECORDER = r"""python scripts\windows\record_episode_150_windows_v7.py `
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
    rows=load_csv(Path("data/protocol/day25_gripper_collection_plan.csv"))
    for i,row in enumerate(rows,1):
        label=" repeat" if row["repeat_slot"]=="true" else ""
        print("="*72)
        print(f"[{i:02d}/20] {row['plan_row_id']}{label}")
        print("方块保持 nominal。正常接近/下降。到 clean 应闭合点时保持夹爪打开。")
        print("不原地等待；沿 nominal clean 抬升方向继续约30–40mm后再闭合。")
        print("不追方块、不移动方块、不加 target offset、不额外偏轨。要求：任务失败。")
        print(RECORDER)
    return 0
if __name__=="__main__":
    raise SystemExit(main())
