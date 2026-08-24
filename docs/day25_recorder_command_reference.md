# Day25 recorder command

```powershell
python scripts\windows\record_episode_150_windows_v7.py `
    --duration 60 `
    --hz 15 `
    --countdown 5 `
    --motion-speed-scale 1.25 `
    --gripper-speed-scale 1.0 `
    --pose configs\poses\follower_home_v2.json `
    --leader-pose configs\poses\leader_episode_home_v1.json `
    --output-dir outputs\episodes_root_cause_v2_final `
    --task "抓取无压纹红色方块并放入固定目标区"
```

Recorder v7 remains frozen unless a reproducible recorder defect is found.
