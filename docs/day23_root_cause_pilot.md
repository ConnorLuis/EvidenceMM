\
# Day23 Excluded Root-Cause Pilot and Intervention Parameter Freeze

## Scope

Day23 executes the pilot required by the frozen Day22 Root-Cause Benchmark v2
protocol. It does not modify the Day22 taxonomy, 90-case layout, pair-group
split rule, or evaluation metrics.

The pilot has exactly 12 protocol rows:

- 3 matched clean controls;
- 3 target-offset interventions;
- 3 gripper-close-timing interventions;
- 3 trajectory-deviation interventions.

All 12 are permanently excluded from final Benchmark v2 metrics.

## Acquisition boundary

Raw acquisition remains in `embodied-agent-arm`.

Reuse the existing Windows recorder:

```powershell
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

Do not place intervention labels in `--task`, raw metadata, image file names,
episode folder names, or model-visible source manifests.

## Preflight gate

Before the 12 pilot rows, run one ordinary clean acquisition as an excluded
preflight.

Proceed only if the recorder itself reports:

```text
OVERALL EPISODE: PASS
```

A run that produces all files but ends with recorder `FAIL` is a technical
exclusion and must not count as a clean control or causal pilot.

## Three intensity levels

Pilot groups are ordered:

```text
G01 = mild
G02 = medium
G03 = strongest_safe
```

For each physical cause, choose one safe intervention direction before its first
pilot row and keep that direction fixed across G01/G02/G03.

The numerical magnitude must increase with intensity rank.

Day22 deliberately did not prescribe universal millimeter/frame values. Day23
freezes values only after they have been tested on the real setup.

## Target offset

- Choose one table-plane direction that has clearance.
- Measure the object displacement in millimeters.
- Keep that direction for all three target-offset pilot rows.
- Increase the measured displacement from G01 to G03.
- After moving the cube, execute the nominal grasp path toward the original
  nominal marker. Do not visually compensate by chasing the moved cube.
- Do not deliberately alter gripper timing or motion trajectory in the same
  episode.

Record:

```text
parameter_direction
parameter_value
parameter_unit=mm
```

## Gripper-close timing

- Keep object pose nominal.
- Keep the nominal approach path.
- Choose one direction: `early` or `late`.
- Use the same direction in all three groups.
- Increase the intended phase shift from G01 to G03.
- Do not add a trajectory deviation.

Day23 derives a numeric timing proxy from `samples.csv`:

```text
first_major_gripper_transition
-
first_sustained_arm_motion
```

and subtracts the paired clean-control phase in the same pilot group.

The operator must inspect the episode and set:

```text
gripper_transition_verified_as_grasp_close=true
```

only when the detected major gripper transition is actually the grasp-close
event.

The frozen numeric unit is:

```text
frames_vs_pair_clean_motion_aligned
```

## Trajectory execution deviation

- Keep object pose nominal.
- Keep gripper timing nominal.
- Choose one free-space lateral deviation direction.
- Mark/measure the lateral waypoint offset in millimeters.
- Keep direction fixed across G01/G02/G03.
- Increase the measured waypoint deviation from G01 to G03.
- Introduce only that bounded path deviation during the approach.
- Never aim the deviation toward fixtures, the table, or a hard collision.

Record:

```text
parameter_direction
parameter_value
parameter_unit=mm
```

## Record contract

After every episode, fill the matching row in:

```text
data/protocol/day23_pilot_records.csv
```

Administrative labels in this CSV are never model input.

Important fields:

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

For a clean control:

```text
intervention_predeclared=false
intervention_applied=false
single_primary_intervention=true
parameter fields blank
```

For a controlled intervention:

```text
intervention_predeclared=true
intervention_applied=true
single_primary_intervention=true
```

## Observability review

`changed_factor_observable` must be judged without using the intervention label
as model evidence.

Allowed evidence:

- front images;
- wrist images;
- observation;
- action;
- tracking_error.

`observable_modalities` is a semicolon-separated subset, for example:

```text
front;wrist
action
front;action
```

## Analysis

After all 12 records are filled:

```bash
python scripts/analyze_day23_root_cause_pilot.py
```

This validates raw episode structure and writes:

```text
data/protocol/day23_pilot_analysis.json
```

For gripper timing it derives the model-visible timing proxy and the signed shift
relative to the clean control in the same group.

## Freeze

If analysis is structurally complete:

```bash
python scripts/freeze_day23_intervention_parameters.py
```

The freeze script requires:

- all 12 recorder runs technically pass;
- 3/3 clean controls succeed;
- at least 2/3 task failures for each physical cause;
- no safety abort or hardware fault;
- interventions are predeclared, actually applied, and single-primary;
- selected parameter row has observable evidence;
- parameter directions are consistent per cause;
- intensity magnitudes are ordered mild < medium < strongest-safe;
- gripper derived timing shifts have the declared early/late sign;
- the selected parameter is an actually tested successful pilot value.

Selection is frozen in advance:

1. prefer the successful medium row;
2. otherwise choose the eligible row nearest to medium;
3. tie -> lower intensity.

No interpolation is allowed.

It writes:

```text
data/protocol/day23_intervention_parameters.json
```

## Final validation

```bash
python scripts/validate_day23_root_cause_pilot.py
```

Day23 is accepted only when it prints `valid=true`.

## Non-goals

Day23 does not:

- train Qwen;
- tune retrieval;
- modify the Day19/Day20 localizer;
- create a held-out split;
- add Agent/MCP;
- put intervention labels into model-visible evidence;
- bypass any robot safety guard.

Day24 may begin only after Day23 is `CLOSED / FROZEN`.
