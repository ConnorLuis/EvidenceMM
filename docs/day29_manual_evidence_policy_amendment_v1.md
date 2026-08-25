# Day29 Manual Evidence Policy Amendment v1

## Status

This amendment is defined before Pass A human review begins.

It does not modify or overwrite the frozen Day29 causal-review SOP or the
frozen review-record schema. It supersedes only the unconditional interpretation
of the answerable-failure manual-reference requirement.

## Reason for amendment

The frozen Day29 review SOP requires manual evidence to be human-verified as
relevant to the claimed causal mechanism and explicitly states that retrieval
rank alone is not causal support.

The frozen review schema also lists `supporting_manual_refs` as a requirement
for every answerable failure.

The only currently approved component manual is the frozen STS3215 datasheet.
Claim-level inspection shows that this source does not provide causal
documentation for visual target localization or perception error, and only
partially documents actuator-level facts relevant to gripper timing and
trajectory execution.

Requiring a non-empty manual reference for every answerable cause would
therefore either force irrelevant citations or incorrectly make manual coverage
determine evidence answerability.

Both outcomes are prohibited.

## Superseding requirement

For an `answerable` failed episode, the reviewer must provide:

- a smallest defensible `failure_interval`;
- one or more `supporting_robot_refs`;
- `supporting_manual_refs` only when the approved manual corpus contains
  human-verified content that is genuinely relevant to a mechanistic claim used
  in the diagnosis.

When no approved manual content is applicable to the claimed mechanism:

- `supporting_manual_refs` must remain empty;
- no unrelated page may be cited merely to satisfy the field;
- `blind_review_notes` must state
  `manual_support_not_applicable_to_claim`.

An empty manual-reference list under this condition does not by itself make the
episode `insufficient_evidence`.

Evidence answerability remains determined by whether the frozen model-visible
evidence distinguishes one benchmark physical cause from the competing causes.

## Current STS3215 coverage

### target_offset_or_perception

Current manual coverage: `not_supported`.

The STS3215 datasheet does not document cameras, visual perception, target
localization, coordinate transforms, target estimates, or target-offset
semantics.

STS3215 pages must not be cited as causal support for a target/perception claim
solely because they are the available approved manual pages.

### gripper_close_timing

Current manual coverage: `partially_supported`.

Pages 3, 4, and 8 may support narrowly stated actuator-level premises such as
servo speed, position-control semantics, feedback, resolution, and maximum
position-update rate.

They do not independently establish that a gripper closed too early or too late
relative to the object-contact window.

The task-level timing diagnosis must come from model-visible robot evidence.

### trajectory_execution_deviation

Current manual coverage: `partially_supported`.

Pages 3, 4, and 8 may support narrowly stated actuator-level premises involving
control capability, position sensing, feedback, resolution, speed, or update
rate.

They do not define the robot-level desired trajectory, an acceptable
tracking-error threshold, or the benchmark definition of trajectory deviation.

The task-level trajectory diagnosis must come from model-visible robot evidence.

## Page 5 limitation

Page 5 documents protection modes and protection behavior.

Those facts must not be used to infer that a protection event occurred unless
the frozen model-visible evidence directly exposes evidence sufficient to make
that claim.

Unobserved current, voltage, temperature, or internal protection flags must not
be inferred.

## Unchanged boundaries

This amendment does not:

- add any episode-specific answer;
- reveal pair-group information;
- reveal collection interventions;
- reveal physical-cause ground truth;
- alter the three physical-cause taxonomy;
- alter Pass A / Pass B separation;
- materialize or use the Day30 development/held-out split;
- authorize any new manual source;
- modify the frozen official blind-review pack.

The official blind-review pack remains the materialization frozen before human
review.
