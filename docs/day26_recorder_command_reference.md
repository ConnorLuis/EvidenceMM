# Day26 Recorder Command Reference

All Day26 episodes use the same anti-label-leakage task text.

```powershell
python scripts\windows\record_episode_150_windows_v7.py `
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
  --task "抓取无压纹红色方块并放入固定目标区"
```

Generate the 20 row-specific prompts with:

```bash
python scripts/generate_day26_recorder_commands.py
```

After each episode, do not register it until the recorder ends with `OVERALL EPISODE: PASS` and you have confirmed task outcome plus the single frozen trajectory intervention.
