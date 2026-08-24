# Day26：Final `trajectory_execution_deviation` collection

## Scope

Day26 only collects the final trajectory-deviation condition. It does not run a new pilot, does not tune the frozen magnitude, does not recollect clean controls, does not create the Day30 split, and does not perform causal Ground Truth review.

## Frozen intervention

- physical cause: `trajectory_execution_deviation`
- intervention type: `manual_bounded_trajectory_deviation`
- primary changed factor: `commanded_motion_path`
- direction: `follower_forward`
- magnitude: `40–60 mm`
- operational measurement: `marked_lateral_waypoint_offset`
- measurement precision: operator-estimated range

The 40–60 mm range is the Day23 medium pilot parameter and is frozen for Day26. Do not increase or decrease it based on individual outcomes.

## Operator procedure

1. Keep the red cube at the nominal marker.
2. Keep gripper timing nominal.
3. Approach normally until the end effector is near the normal above-object approach region.
4. Through Leader teleoperation, introduce one bounded commanded-path deviation toward Follower forward by approximately 40–60 mm.
5. Maintain that deviation while descending.
6. Close the gripper at the normal clean timing.
7. Continue the episode without adding a second primary intervention.
8. Do not move the object, do not shift gripper timing, and do not compensate back toward the cube.
9. The controlled intervention episode is required to fail the task.

## Canonical plan

- 15 primary trajectory slots: all `rcv2_g01_s04` through `rcv2_g15_s04`.
- 5 trajectory repeat slots: `rcv2_g03_s05`, `rcv2_g06_s05`, `rcv2_g09_s05`, `rcv2_g12_s05`, `rcv2_g15_s05`.
- 20 new episodes total.
- Day24 `s01` clean anchors are reused 15/15.

## Experimental eligibility

A Day26 attempt is canonical only when:

- recorder technical audit passes;
- task fails;
- the declared trajectory intervention was applied;
- it is the single primary intervention;
- the changed commanded path is observable;
- the operator verifies the frozen 40–60 mm forward deviation proxy was met;
- no safety abort or hardware fault occurred.

A successful trajectory episode is an experimental exclusion and the same plan row must be recollected with the same frozen parameter.

## Anti-label leakage

The recorder task text remains identical across all physical causes:

`抓取无压纹红色方块并放入固定目标区`

Do not write cause labels, plan-row IDs, pair-group IDs, intervention parameters, or task-success Ground Truth into raw Episode metadata.

## Day26 close condition

Day26 closes only after:

- trajectory canonical = 20/20;
- trajectory failure = 20/20;
- Day24 clean anchors = 15/15;
- complete groups = 15/15;
- all 20 new canonical episodes pass recorder v7 technical audit;
- future split remains unmaterialized;
- targeted tests and full tests pass;
- Day26 validator passes;
- commit/push/remote verification completes.
