\
# Day23：12 条 Pilot 具体操作清单（中文）

## 0. 先做一个额外的预检，不算 12 条 Pilot

你现在的数据采集入口已经能完成：

```text
Home
→ Leader 对齐
→ 900/900 固定采样
→ 双相机
→ samples.csv
→ Home / Park
→ 扭矩关闭
```

但 Day23 的要求不是“脚本能跑完”，而是录制器最后必须：

```text
OVERALL EPISODE: PASS
```

所以在正式 12 条 Pilot 前，先按正常抓取再录一条 **额外 clean preflight**。

这条预检：

```text
不属于 12 条 Pilot
不进入最终 Benchmark
```

只有它 `OVERALL EPISODE: PASS` 后，才开始下面 12 条。

如果出现：

```text
wrist_fps: FAIL
wrist_duplicate_ratio: FAIL
OVERALL EPISODE: FAIL
```

就先处理采集技术问题，不要继续批量录 Pilot。

---

## 1. 固定采集命令

在 Windows PowerShell：

```powershell
cd C:\Users\Administrator\projects\embodied-agent-arm

python scripts\windows\record_episode_150_windows_v5.py `
    --duration 60 `
    --hz 15 `
    --countdown 5 `
    --motion-speed-scale 1.25 `
    --gripper-speed-scale 1.0 `
    --pose configs\poses\follower_home_v2.json `
    --leader-pose configs\poses\leader_episode_home_v1.json `
    --output-dir outputs\episodes_root_cause_v2_pilot_day23 `
    --task "抓取无压纹红色方块并放入固定目标区"
```

12 条都保持这套采集参数。

**不要**把 `target_offset`、`gripper_timing` 等标签写进 `--task`。

---

## 2. 开始前先做三件事

### A. 给正常红色方块起点做一个桌面标记

这是：

```text
nominal object marker
```

clean、gripper、trajectory 都从这个位置开始。

target-offset 才移动方块。

### B. 给正常抓取轨迹形成一个稳定操作习惯

clean 时尽量每次：

```text
Home
→ 接近红色方块
→ 对准
→ 闭合夹爪
→ 抬起
→ 移动到目标区
→ 放下
```

不用追求机器人轨迹每帧完全一致，但不要随意增加多余动作。

### C. 为三个 intervention 分别选一个安全方向

正式开始前确定：

```text
target offset：选一个桌面平面方向
gripper timing：只选 early 或只选 late
trajectory deviation：选一个朝自由空间的横向方向
```

同一 physical cause 的 G01/G02/G03 **方向必须一致**。

不要 G01 左偏、G02 右偏。

---

# 3. 第一组 G01：mild

## ① p23_g01_clean

目标：

```text
正常成功抓取
```

操作：

1. 红色方块放回 nominal marker。
2. 不施加任何 intervention。
3. 运行采集命令。
4. Leader 按正常方式控制 Follower 完成抓取和放置。
5. 尽量成功完成任务。
6. 等待脚本自动 Home → Park → 关闭扭矩。
7. 记下 episode 文件夹名。

记录：

```text
task_success=true
intervention_predeclared=false
intervention_applied=false
single_primary_intervention=true
```

---

## ② p23_g01_target —— mild target offset

目标：

```text
制造轻度 target_offset_or_perception
```

操作：

1. 保持机器人、相机和目标区不变。
2. 将红色方块从 nominal marker 向你事先选定的安全方向移动一个**轻度、可测量**距离。
3. 用尺子量实际移动距离，记录 mm。
4. 运行采集。
5. **关键：仍按照原 nominal marker 的正常抓取路径操作，不要因为你看到方块移动了就追着方块修正。**
6. 不要故意提前/延迟闭合夹爪。
7. 不要额外做横向轨迹偏移。
8. 观察是否因此导致抓取任务失败。

记录：

```text
parameter_direction=<固定方向>
parameter_value=<实际毫米数>
parameter_unit=mm
intervention_applied=true
single_primary_intervention=true
```

---

## ③ p23_g01_gripper —— mild gripper timing shift

目标：

```text
制造轻度 gripper_close_timing
```

操作：

1. 方块必须放回 nominal marker。
2. 正常接近方块。
3. 保持机械臂路径尽量与 clean 相同。
4. 按你预先确定的唯一方向：
   - early：比正常抓取阶段更早闭合；
   - 或 late：比正常抓取阶段更晚闭合。
5. G01 只做轻度 timing shift。
6. 除夹爪闭合时机外，不要故意改变路径。
7. 采集后检查 front/wrist，确认脚本后续识别出的 major gripper transition 真的是“抓取闭合”动作。

记录：

```text
parameter_direction=early 或 late
parameter_value 留空
parameter_unit 留空
gripper_transition_verified_as_grasp_close=true/false
```

具体 frames shift 由 Day23 分析脚本从 `samples.csv` 和同组 clean 自动计算。

---

## ④ p23_g01_trajectory —— mild trajectory deviation

目标：

```text
制造轻度 trajectory_execution_deviation
```

操作：

1. 方块放回 nominal marker。
2. 闭合夹爪的时机保持正常。
3. 选一个**朝自由空间**的横向方向。
4. 在接近抓取位置的过程中，故意让末端经过一个轻度偏离 nominal approach line 的 waypoint。
5. 这个偏移不要朝桌面、硬件、墙、支架或其他障碍物。
6. 用桌面标记/尺子记录你计划的横向 waypoint 偏移量（mm）。
7. 不要同时移动方块。
8. 不要故意改变夹爪闭合时机。

记录：

```text
parameter_direction=<固定自由空间方向>
parameter_value=<实际标记毫米数>
parameter_unit=mm
```

---

# 4. 第二组 G02：medium

先重新录 clean，再分别做三类 medium intervention。

## ⑤ p23_g02_clean

与 G01 clean 相同。

必须尝试正常成功。

---

## ⑥ p23_g02_target

规则与 G01 target 完全相同，但：

```text
G02 offset mm > G01 offset mm
```

方向不变。

---

## ⑦ p23_g02_gripper

规则与 G01 gripper 相同，但：

```text
timing shift 强度 > G01
```

early/late 方向不变。

---

## ⑧ p23_g02_trajectory

规则与 G01 trajectory 相同，但：

```text
G02 waypoint offset mm > G01
```

方向不变。

---

# 5. 第三组 G03：strongest-safe

`strongest-safe` 不是“越极端越好”。

它的含义是：

> 在现有安全保护、无硬碰撞、无硬件风险、工作空间有充分余量的条件下，你愿意重复用于正式数据采集的最大 Pilot 强度。

## ⑨ p23_g03_clean

正常抓取。

---

## ⑩ p23_g03_target

```text
G03 target offset > G02 > G01
```

方向不变。

不追着移动后的方块修正 nominal path。

---

## ⑪ p23_g03_gripper

```text
|G03 timing shift| > |G02| > |G01|
```

方向仍然保持 early 或 late 中最初选的那个。

---

## ⑫ p23_g03_trajectory

```text
G03 waypoint offset > G02 > G01
```

仍然只朝最初选定的自由空间方向。

---

# 6. 每录完一条立即填记录

文件：

```text
data/protocol/day23_pilot_records.csv
```

最少要填写：

```text
episode_id
raw_episode_relpath
recorder_overall_pass
failed_checks
task_success
intervention_predeclared
intervention_applied
single_primary_intervention
parameter_direction
parameter_value
parameter_unit
changed_factor_observable
observable_modalities
gripper_transition_verified_as_grasp_close
safety_abort
hardware_fault
operator_notes
```

### raw_episode_relpath

例如：

```text
20260823_101530
```

因为 Day23 配置已经固定 dataset root，所以只写 episode 文件夹名即可。

### recorder_overall_pass

只根据采集器最后：

```text
OVERALL EPISODE: PASS / FAIL
```

填写。

如果 FAIL：

```text
recorder_overall_pass=false
failed_checks=wrist_fps;wrist_duplicate_ratio
```

### observable_modalities

使用分号：

```text
front
wrist
action
tracking_error
front;wrist
front;action
wrist;action;tracking_error
```

---

# 7. 什么情况立刻停止该条

出现任意一个：

```text
意外硬接触
异常掉电
机械臂明显失控
安全守卫触发
相机/串口异常
需要关闭安全限制才能继续
```

立即停止。

记录：

```text
safety_abort=true
或
hardware_fault=true
```

这条保留，但不能成为可接受 Pilot。

---

# 8. Day23 最终不是要求 9/9 intervention 全部失败

冻结标准是：

```text
3 clean：
3/3 技术 PASS 且任务成功

target offset：
至少 2/3 技术 PASS + intervention 正确施加 + 任务失败

gripper timing：
至少 2/3

trajectory deviation：
至少 2/3
```

如果某一类只有：

```text
1/3
```

则 Day23 不通过。

正确做法是重新做该类 Pilot/重新校准参数，不是降低标准。

---

# 9. 12 条完成后的命令

在 EvidenceMM / WSL：

```bash
cd ~/projects/evidencemm

python scripts/analyze_day23_root_cause_pilot.py
```

然后：

```bash
python scripts/freeze_day23_intervention_parameters.py
```

然后：

```bash
python scripts/validate_day23_root_cause_pilot.py
```

最终必须：

```text
valid=true
```

才允许进入 Day24 正式 90-case benchmark 数据采集。

---

# 10. Day23 最重要的原则

你不是在“故意演一个看起来像某类故障的视频”。

你是在做：

```text
提前声明唯一 intervention
→ 真实 Leader/Follower 执行
→ 保留完整 front/wrist/state/action
→ 检查 intervention 是否真的造成 task failure
→ 检查原因变化是否可从允许证据中观察
→ 冻结一个真实测试过的参数
```

这样 Day29 才有资格把 `physical_cause_gt` 当成真正的 causal Ground Truth。
