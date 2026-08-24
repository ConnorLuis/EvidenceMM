# Day25 — final gripper-close-timing collection

Day25 only collects the frozen `gripper_close_timing` cause.

**No new clean episodes are collected.** The 15 selected Day24 `s01` clean episodes are reused as matched clean anchors.

New episodes: 15 primary `s03` + 5 repeats `s05` for `g02/g05/g08/g11/g14` = **20**.

## Frozen intervention

- cause: `gripper_close_timing`
- type: `manual_gripper_close_timing_shift`
- changed factor: `gripper_close_phase`
- direction: `late`
- phase proxy: `upward_progress_after_nominal_close_point_before_close`
- range: **30–40 mm**, operator-estimated

Procedure: keep cube at nominal. Approach and descend normally. At the clean episode's normal close point keep the gripper open. **Do not pause at the grasp pose.** Continue along the nominal clean lifting direction for about 30–40 mm with the gripper still open, then close. Continue the remaining nominal path. Do not move the cube, add target offset, or add extra trajectory deviation.

This is matched manual teleoperation, not an exact autonomous trajectory replay claim.

A technically valid gripper episode is canonical only when the intervention is the single primary intervention, the phase proxy was followed, and the task fails.

Day25 closes at 20/20 gripper failures + 15/15 Day24 clean anchors + 15/15 complete pair groups + all 20 new recorder-v7 technical PASS + no Day30 split.
