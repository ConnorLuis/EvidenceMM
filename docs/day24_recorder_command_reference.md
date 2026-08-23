# Day24 Recorder Command Reference

所有 Episode 使用同一个无标签泄漏 task 文本。每次按清单完成物理操作后执行：

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

生成逐条 35-row 操作提示：

```bash
python scripts/generate_day24_recorder_commands.py
```
