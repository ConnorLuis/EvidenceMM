# Day24：Final `target_offset_or_perception` + matched clean 采集

## 1. Day24 在冻结路线中的位置

Day24 **只做最终 target-offset 数据采集**，不做模型训练、不做 Ground Truth 人工因果审查、不做 held-out split。

冻结路线：

- Day22：Root-Cause Benchmark v2 协议冻结
- Day23：12 条排除型 Pilot；验证三类干预并冻结操作参数
- **Day24：15 个 pair group 的 matched clean + 20 条 target-offset 最终数据**
- Day25：gripper-close-timing 最终数据
- Day26：trajectory-deviation 最终数据
- Day27：insufficient-evidence + 缺口补录
- Day28：raw audit / exclusion / source binding
- Day29：Human causal review + failure interval + robot/manual evidence GT
- Day30：Ground Truth + pair-group development/held-out split 冻结
- Day31：Qwen3-VL root-cause baseline
- Day32：development-only calibration；只有在稳定错误模式证明必要时才考虑 QLoRA
- Day33：一次性 fresh held-out final evaluation
- Day34：最终指标、latency、GPU memory、error analysis、Canonical E2E
- Day35：README / release / 简历 / 面试材料，正式封板

## 2. 为什么 Day24 是 35 条，而不是 30/40/45 条

Day22 已冻结最终 benchmark：

- 15 个 pair groups
- 每组只有 **1 个 clean control**
- clean 可同时作为同组 target / gripper / trajectory 的 matched anchor
- target cause 最终需要 20 条：15 个 primary target + 5 个 rotating repeat target

因此 Day24 一次性采：

- clean：15
- target primary：15
- target repeat：5
- 总计：**35 条 canonical slots**

Day25、Day26 **复用 Day24 的同 pair-group clean**；除非 clean 无效/技术失败，否则不重复采 clean。

## 3. Day23 冻结后 Day24 使用哪个 target 参数

Day23 target Pilot：

- G01：20 mm，failure
- G02：40 mm，failure
- G03：60 mm，failure

Day22 的选择规则优先采用有效的 medium 参数，因此最终 Day24 target 参数冻结为：

> **方块沿 Follower 正前方移动 40 mm。**

所有 20 条 target 都使用同一方向、同一 40 mm 幅度。

如果某条 40 mm target 意外成功：

- 原始 Episode 必须保留；
- 记录为 experimental exclusion；
- **不能临时把参数改成 60 mm**；
- 同一个 plan row 仍使用冻结的 40 mm 重录。

## 4. matched pair 的物理要求

同一 `pair_group_id` 的所有成员必须共享：

- 场景布置；
- 同一个红色无压纹方块；
- nominal 起始标记；
- front/wrist 相机位置；
- recorder v7、60 s、15 Hz、Home/Park 等 acquisition 配置。

Day24 clean 采完后，Day25/26 的同组数据继续引用这个 clean anchor。

## 5. Clean 操作

1. 红色方块放回 nominal 标记框。
2. 正常接近。
3. 正常下降。
4. 正常闭合夹爪。
5. 正常抬升。
6. 正常放入固定目标区。

要求：任务成功。

Clean 失败时：

- 保留 raw attempt；
- 标记 experimental exclusion；
- 排查人为操作后重录相同 plan row。

## 6. Target 操作

1. 先确认该 pair group 的场景与 clean 完全一致。
2. 将方块从 nominal 标记沿 **Follower 正前方移动 40 mm**。
3. 不改变机器人的 nominal 抓取意图。
4. 机械臂仍然走向原来的 nominal 抓取位置。
5. 不追偏移后的方块。
6. 不补偿。
7. 不改变夹爪时序。
8. 不额外加入 trajectory deviation。
9. 完成这一轮操作。

要求：target 干预应造成任务失败。

## 7. 原始数据格式：不是 MP4

EvidenceMM 的机器人 canonical source 是：

- `front/` 同步图像序列；
- `wrist/` 同步图像序列；
- `samples.csv` 中 observation / action / tracking_error / elapsed_ns；
- `metadata.json`。

后续所谓“视频时间定位”在机器人 benchmark 上实现为：

> **同步图像序列上的 temporal slicing / key-frame localization**

无需先转成 MP4，也没有偏离多模态时序证据目标。

## 8. anti-label-leakage

Recorder 的 `--task` 对 clean 和 target 完全相同：

`抓取无压纹红色方块并放入固定目标区`

原始 Episode metadata 里不要写：

- `target_offset_or_perception`
- `pair_group_id`
- `plan_row_id`
- 40 mm 干预标签
- task_success Ground Truth

这些只存在 EvidenceMM 的 admin protocol records 中。

## 9. 技术失败与实验失败必须分开

技术有效：

- `OVERALL EPISODE: PASS`
- samples = 900
- front = 900
- wrist = 900
- frame index 0..899
- elapsed_ns 严格递增
- state/action/tracking_error 有限
- recorder checks 全 PASS

实验有效：

Clean：
- task_success = true
- no intervention

Target：
- task_success = false
- target 40 mm 已实施
- single primary intervention
- changed factor 在 front/wrist 中可观察

任何技术失败均不能被当作 target failure。

## 10. Day24 CLOSED 条件

必须全部满足：

- 35/35 canonical slots；
- 15/15 clean success；
- 20/20 target failure；
- 35/35 recorder technical PASS；
- target 方向全部为 `follower_forward`；
- target 幅度全部为 40 mm；
- 15/15 pair groups complete；
- 所有 excluded/recollection raw attempts 均保留；
- 不生成 Day30 split；
- tests PASS；
- Day24 validator PASS；
- commit/push/remote verify。

达到以上条件后才能标记：

`DAY24 CLOSED / FROZEN`
