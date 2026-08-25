# Day29 Root-Cause Human Review SOP

## Purpose

This SOP defines the generic human-review procedure for EvidenceMM
Root-Cause Benchmark v2.

It is case-agnostic. It must not contain episode IDs, pair-group IDs,
intervention assignments, or benchmark answers.

## Core separation

Three concepts must remain distinct:

1. observed symptom;
2. verified physical cause;
3. whether model-visible evidence is sufficient to diagnose that cause.

A physical cause may be administratively known while the evidence remains
insufficient for diagnosis.

## Pass A — blind evidence review

The reviewer may inspect only:

- front images;
- wrist images;
- observation;
- action;
- tracking_error;
- approved manual corpus.

The reviewer must not inspect:

- pair_group_id;
- plan_row_id;
- planned physical cause;
- intervention type;
- intervention parameters;
- intervention-applied metadata;
- physical_cause_gt;
- evidence_answerability_gt;
- diagnostic_decision_gt;
- prior human-review answers.

Pass A records:

- task symptom;
- smallest defensible failure interval;
- supporting robot evidence;
- counterevidence robot evidence;
- supporting manual evidence;
- counterevidence manual evidence;
- answerability judgment;
- explicit uncertainty reason when evidence is insufficient;
- blind-review confidence;
- blind-review notes.

## Failure interval

For a failed episode, select the smallest defensible interval containing the
first observable transition from the nominal task progression into the failure.

Both cameras and state/action signals should be inspected.

Do not enlarge an interval merely because surrounding frames are convenient.

For clean-success episodes, failure_interval must be null.

## Evidence answerability

Use `answerable` only when model-visible evidence supports one physical cause
well enough to distinguish it from the other benchmark causes.

Use `insufficient_evidence` when the failure is observable but the available
evidence cannot uniquely support a physical-cause diagnosis.

Use `not_applicable_clean` for verified clean-success episodes.

Technical corruption is never an insufficient-evidence case.

## Robot evidence references

Robot evidence references should identify defensible source locations using
canonical EvidenceRef-style fields such as:

- source_id;
- source_type;
- frame_index;
- time_start_sec;
- time_end_sec;
- camera;
- optional note.

A supporting reference must directly support the review statement.
A counterevidence reference must weaken a plausible competing hypothesis.

## Manual evidence

Manual evidence is causal support only when the cited manual content is
human-verified as relevant to the claimed mechanism.

Retrieval rank alone is not causal support.

Manual references should identify:

- source_id;
- page_number;
- optional supporting text summary;
- relevance note.

The current approved component source is the frozen STS3215 datasheet.

## Pass B — administrative reveal

Only after Pass A is complete may the reviewer inspect administrative
collection metadata.

Pass B verifies:

- pair_group_id;
- technical_valid;
- experimental_valid;
- task_success;
- intervention_verified;
- physical_cause_gt.

Pass B must not retroactively change the Pass-A answerability judgment merely
because the intervention becomes known.

## Diagnostic decision

Decision rule:

- verified clean success -> `clean_success`;
- failed case with insufficient model-visible evidence ->
  `insufficient_evidence`;
- otherwise diagnostic_decision_gt must equal the verified physical_cause_gt.

Allowed physical causes:

- target_offset_or_perception;
- gripper_close_timing;
- trajectory_execution_deviation.

## Confidence

Confidence expresses confidence in the human annotation, not model confidence.

Use a numeric value in [0, 1].

Low confidence should be explained in review_notes.

## Review completion

An answerable physical-cause failure requires:

- failure_interval;
- supporting_robot_refs;
- supporting_manual_refs.

An insufficient-evidence failure requires:

- failure_interval;
- explicit_uncertainty_reason.

A clean success requires:

- no failure_interval;
- evidence_answerability_gt = not_applicable_clean;
- diagnostic_decision_gt = clean_success.

## Prohibited operations

Day29 must not:

- materialize development/held-out membership;
- use future split information;
- tune retrieval, prompts, or models against future held-out membership;
- overwrite Day24-Day28 frozen records;
- convert technical corruption into insufficient evidence.
