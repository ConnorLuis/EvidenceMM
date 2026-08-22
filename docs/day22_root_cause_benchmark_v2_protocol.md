# Day22 Root-Cause Benchmark v2 Protocol Freeze

## 1. Roadmap decision

Day22 does not add a generator or a new localizer.

After Day21, the missing flagship capability is still the physical robot
root-cause target defined by EvidenceMM:

```text
target offset / perception error
gripper-close timing error
trajectory execution deviation
insufficient evidence
```

Day21 already established:

```text
localized robot evidence
+
front / wrist
+
observation / action / tracking_error
+
manual retrieval
+
UnifiedEvidenceBundle
+
explicit abstention
```

but correctly refused physical-cause claims because there is no discriminative
causal supervision or human-verified manual causal support.

The project route is therefore extended to Day35:

```text
Day22  protocol freeze
Day23  excluded pilot + intervention-parameter freeze
Day24  target-offset final collection
Day25  gripper-timing final collection
Day26  trajectory-deviation final collection
Day27  insufficient-evidence + clean controls + recollection
Day28  raw audit / exclusions / source binding
Day29  Human causal review + manual-support annotation
Day30  GT + groupwise development/held-out split freeze
Day31  root-cause diagnostic baseline
Day32  development-only calibration
Day33  one frozen held-out final evaluation
Day34  metrics / error analysis / latency / GPU / E2E
Day35  final documentation / release / interview closure
```

This remains the original EvidenceMM flagship route rather than a new project.

## 2. Project ownership

Robot control and raw acquisition remain in:

```text
embodied-agent-arm
```

The existing Leader -> Follower acquisition path is reused. Day22 requires no
new acquisition controller in EvidenceMM.

EvidenceMM owns:

```text
benchmark protocol
source binding
Ground Truth
retrieval
temporal localization
root-cause diagnosis
grounding
evaluation
```

Do not copy robot-control logic into EvidenceMM.

## 3. Why controlled intervention is required

A visible failure symptom is not a physical cause.

For example:

```text
object dropped
!=
gripper_close_timing
```

and:

```text
large tracking error
!=
trajectory_execution_deviation
```

unless the causal source is independently known.

Root-Cause Benchmark v2 therefore uses:

```text
predeclared controlled intervention
        ->
known changed factor
        ->
real robot episode
        ->
human evidence review
        ->
causal Ground Truth
```

The intervention metadata is administrative Ground Truth and is never available
to the diagnostic model.

## 4. Frozen causal taxonomy

Physical causes:

```text
target_offset_or_perception
gripper_close_timing
trajectory_execution_deviation
```

Administrative non-causal physical values:

```text
unknown
none_clean
```

Evidence answerability:

```text
answerable
insufficient_evidence
not_applicable_clean
```

Final diagnostic decisions:

```text
target_offset_or_perception
gripper_close_timing
trajectory_execution_deviation
insufficient_evidence
clean_success
```

The key distinction is:

```text
physical_cause_gt
!=
evidence_answerability_gt
```

A controlled intervention can establish:

```text
physical_cause_gt = target_offset_or_perception
```

while a reviewer may still conclude:

```text
evidence_answerability_gt = insufficient_evidence
diagnostic_decision_gt = insufficient_evidence
```

if front/wrist/state/action/manual evidence cannot verify that cause.

This is the benchmark's abstention Ground Truth.

## 5. Symptom and cause remain separate

The Day16 lesson is preserved.

Human review must keep:

```text
observed failure symptom
```

separate from:

```text
physical cause
```

Do not convert symptoms such as `grasp_drop` or `post_place_collision` directly
into causal labels.

## 6. Day23 pilot: 12 episodes, excluded from benchmark

Before final collection, Day23 collects three pilot pair groups.

Each pilot group contains:

```text
1 clean control
1 target-offset intervention
1 gripper-timing intervention
1 trajectory-deviation intervention
```

Total:

```text
3 groups x 4 episodes = 12 pilot episodes
```

All 12 are permanently excluded from Root-Cause Benchmark v2 final metrics.

Pilot purposes:

1. confirm the intervention can be executed through the existing Leader
   teleoperation path;
2. confirm the intervention remains inside the existing safe workspace;
3. establish numeric intervention magnitudes/directions;
4. confirm at least 2/3 pilot interventions per causal class induce the intended
   task failure;
5. require all three clean controls to succeed and remain technically valid;
6. confirm the changed factor is observable in at least one permitted evidence
   stream without using intervention metadata.

At the end of Day23, the numeric parameter policy is frozen.

Day22 intentionally does **not** invent millimeter, frame, angle, speed, or
force values before that pilot.

## 7. Final collection structure: 90 eligible target episodes

The final benchmark is organized into 15 pair groups.

Each pair group has six planned slots:

```text
slot 1  clean control
slot 2  target_offset_or_perception
slot 3  gripper_close_timing
slot 4  trajectory_execution_deviation
slot 5  rotating repeated causal class
slot 6  insufficient_evidence candidate
```

The rotating slot follows:

```text
group 1: target
group 2: gripper
group 3: trajectory
repeat every three groups
```

Across 15 groups:

```text
clean controls                         15
insufficient-evidence candidates       15
target_offset_or_perception            20
gripper_close_timing                   20
trajectory_execution_deviation         20
                                      ---
total                                  90
```

`data/protocol/day22_root_cause_collection_plan.csv` is the frozen run sheet.

The `episode_id` fields are intentionally blank until real acquisition.

## 8. Pair-group protocol

A pair group is a causal comparison block.

Members should keep the following stable as far as practical:

```text
scene setup
object identity
nominal start condition
camera setup
acquisition configuration
```

Only the predeclared primary factor changes for a controlled-cause episode.

One clean control may anchor multiple intervention slots in its pair group.

All six members of a pair group must later remain in the same development or
held-out split.

This avoids train/test leakage through near-counterfactual paired episodes.

## 9. Controlled intervention definitions

### 9.1 Target offset / perception

Frozen intervention type:

```text
object_target_pose_offset
```

Primary changed factor:

```text
environment_target_pose
```

The object/target-relative pose is changed while nominal robot-operation intent
is kept unchanged.

Day23 determines safe numeric offset magnitudes.

### 9.2 Gripper-close timing

Frozen intervention type:

```text
manual_gripper_close_timing_shift
```

Primary changed factor:

```text
gripper_close_phase
```

Through existing Leader teleoperation, the gripper is intentionally closed
earlier or later than the nominal phase.

Do not simultaneously introduce a trajectory intervention.

Day23 determines the operational timing-shift parameterization.

### 9.3 Trajectory execution deviation

Frozen intervention type:

```text
manual_bounded_trajectory_deviation
```

Primary changed factor:

```text
commanded_motion_path
```

The object condition remains nominal while one bounded, predeclared motion-path
deviation is introduced through Leader teleoperation.

Day23 determines a safe magnitude/direction policy.

## 10. Single-primary-intervention rule

Every controlled-cause episode must declare exactly one primary intervention
before acquisition.

If two primary causal factors are deliberately changed in one episode:

```text
experimental_exclusion
```

Do not retrospectively choose whichever label makes the episode convenient.

## 11. Failed intervention attempts are not silently deleted

Every raw attempt is retained in the audit.

Examples:

```text
declared intervention not actually applied
controlled intervention did not produce the required failure
clean control failed unexpectedly
pair setup was no longer comparable
```

These become experimental exclusions or trigger recollection.

The final benchmark targets 90 eligible episodes, but raw attempt count may be
larger.

Final reporting must include intervention failure-induction rate and exclusion
counts rather than hiding unsuccessful attempts.

## 12. Insufficient evidence is not technical corruption

The `insufficient_evidence` decision requires a technically valid failure.

Do **not** use:

```text
missing camera
corrupt CSV
power loss
duplicate/static camera failure
```

as "insufficient evidence."

Those are technical exclusions.

An insufficient-evidence case means:

> the permitted evidence is technically valid but does not uniquely support a
> specific physical cause.

The physical cause may be known from the administrative intervention log or may
remain unknown. The model must still abstain if the model-visible evidence is
not sufficient.

## 13. Technical exclusions

Examples include:

```text
missing metadata/samples
missing front/wrist stream
frame/timestamp binding failure
severe duplicate/static camera failure
uncontrolled power loss or collapse
safety-stop-aborted episode
```

They remain in the raw audit and never become benchmark positives.

## 14. Future Human Review v2 contract

A verified final record must contain:

```text
episode_id
pair_group_id

technical_valid
experimental_valid
task_success
intervention_verified

physical_cause_gt
evidence_answerability_gt
diagnostic_decision_gt

failure_interval

supporting_robot_refs
counterevidence_robot_refs

supporting_manual_refs
counterevidence_manual_refs

confidence
review_notes
```

For an answerable physical cause:

```text
failure interval required
supporting robot refs required
supporting manual refs required
diagnostic_decision_gt == physical_cause_gt
```

For insufficient evidence:

```text
failure interval required
explicit uncertainty reason required
diagnostic_decision_gt = insufficient_evidence
```

For clean control:

```text
task_success = true
physical_cause_gt = none_clean
evidence_answerability_gt = not_applicable_clean
diagnostic_decision_gt = clean_success
failure_interval = null
```

The module freezes these semantics now; the actual Day29 review tooling will
reuse the contract.

## 15. Manual causal Ground Truth

The existing STS3215 datasheet remains valid component evidence, but retrieval
rank alone is not causal support.

Before final Human Review, the project must add/freeze a generic operation and
failure-diagnosis SOP covering evidence rules such as:

```text
nominal pick-place phases
target-pose tolerance concepts
gripper-close phase
trajectory-following interpretation
tracking-error interpretation
evidence required for a cause claim
counterevidence
when to abstain
```

The SOP must be generic:

```text
no test episode IDs
no case answers
no per-episode diagnosis table
```

Human reviewers then annotate which manual pages truly support a diagnosis.

This enables later manual Recall@K and citation precision/recall.

## 16. Anti-label-leakage contract

Model-visible evidence is limited to:

```text
front images
wrist images
observation
action
tracking_error
approved manual corpus
```

Administrative fields that must never enter model prompts, retrieval documents,
source manifests, or model-visible metadata include:

```text
pair_group_id
plan_row_id
planned_physical_cause
intervention_type
intervention parameters
intervention_applied
physical_cause_gt
evidence_answerability_gt
diagnostic_decision_gt
human review notes
Ground Truth supporting refs
```

The collection plan is an administrative run sheet, not an EvidenceMM retrieval
source.

## 17. Fresh future split

The old Day20 held-out set is already exposed and cannot be reused for new model
selection claims.

Root-Cause Benchmark v2 therefore creates a fresh split at **pair-group level**.

Frozen rule:

```text
unit:
pair_group_id

seed:
evidencemm-root-cause-v2-split-v3

rank:
SHA256(seed | pair_group_id), ascending

development:
10 pair groups = 60 planned eligible episodes

held-out:
5 pair groups = 30 planned eligible episodes
```

Day22 freezes the rule but intentionally does not write membership.

Membership is materialized on Day30 only after the final eligible pair groups
and Ground Truth are frozen.

No pair group may cross the split.

No held-out data may be used for:

```text
prompt tuning
retrieval tuning
abstention-threshold tuning
model selection
```

## 18. Final metrics frozen on Day22

### Temporal

```text
Event Recall
Mean Best IoU
Recall@IoU 0.25
Recall@IoU 0.50
Onset absolute error
Offset absolute error
```

### Physical diagnosis

On answerable failed cases:

```text
3-class cause Macro-F1
```

On all failed cases:

```text
4-way diagnostic-decision Macro-F1
target_offset_or_perception
gripper_close_timing
trajectory_execution_deviation
insufficient_evidence
```

Also:

```text
Abstention Accuracy
False Answer Rate
False Abstention Rate
Clean-Control False Positive Cause Rate
```

### Grounding

```text
Robot Citation Precision / Recall
Manual Citation Precision / Recall
Manual Recall@K
Out-of-Bundle Citation Rate
```

### Efficiency

```text
End-to-End Latency
Peak GPU Memory
```

## 19. Safety boundary

Day22 does not alter the robot safety policy.

All causal data collection must preserve the existing safe Leader/Follower
teleoperation envelope.

Frozen rules:

```text
do not bypass safety limits
do not disable existing safety guards
do not manufacture failure by deliberate power loss
do not use hard collision as an intervention
keep intervention inside the existing safe workspace
abort and exclude unexpected contact or hardware fault
```

The intended failure is task failure, not hardware damage.

## 20. Day22 acceptance

Day22 is complete when:

```text
pytest: all pass

build:
protocol_status =
root_cause_benchmark_v2_protocol_frozen_pre_collection

collection plan:
15 pair groups
90 target eligible rows
20 target-offset slots
20 gripper-timing slots
20 trajectory-deviation slots
15 insufficient-evidence candidate slots
15 clean-control slots

pilot:
12 episodes
excluded from final benchmark

future split:
membership materialized = false
held-out model selection = false

numeric intervention values:
not frozen on Day22
must be frozen after Day23 pilot

validator:
valid = true

git diff --check:
clean
```

After commit, Day22 becomes:

```text
CLOSED / FROZEN
```

Day23 may determine safe numeric intervention parameters using only the excluded
pilot. Changing the Day22 taxonomy, 90-case layout, leakage rules, paired-control
contract, or split rule after final collection begins requires an explicit
benchmark protocol version bump.
