# Day29 Pass A Operator Guide

## Purpose

Pass A is a fully blind human review of all 90 canonical benchmark episodes.

Do not inspect collection-plan rows, pair-group assignments, intervention
metadata, task-success metadata, physical-cause ground truth, or any future
development/held-out split.

## Allowed evidence

Only inspect:

- front camera images;
- wrist camera images;
- observation;
- action;
- tracking_error;
- the frozen approved STS3215 manual corpus.

## Review order

Always use the frozen randomized review order contained in the official Day29
blind-review pack.

Do not reorder cases by episode ID.

## Required blind judgment

For every case record:

1. describe the observed task symptom;
2. select the smallest defensible failure interval when a failure is visible;
3. record supporting robot evidence;
4. record counterevidence where relevant;
5. use manual references only when genuinely relevant;
6. decide evidence answerability;
7. when answerable, record one unique blind physical-cause hypothesis;
8. record confidence and notes.

The audit-only `blind_cause_hypothesis` field is retained so that Pass B can
measure whether an answerable blind diagnosis agreed with the administratively
verified physical cause.

It does not alter the frozen final review-record schema.

## Answerability

`answerable`:

- a failure is visible;
- the model-visible evidence supports one unique benchmark cause strongly enough
  to distinguish it from the alternatives;
- `blind_cause_hypothesis` must contain exactly that cause.

`insufficient_evidence`:

- a failure is visible;
- the evidence does not uniquely support one benchmark cause;
- `blind_cause_hypothesis` must be null;
- `explicit_uncertainty_reason` is mandatory.

`not_applicable_clean`:

- no task failure is observed;
- failure interval must be null;
- blind cause must be null.

## Manual evidence

Follow the frozen Day29 manual-evidence amendment.

The current STS3215 source does not support target/perception causal claims.

For target/perception diagnoses:

- supporting manual references must remain empty;
- blind notes must include
  `manual_support_not_applicable_to_claim`.

For gripper-timing and trajectory diagnoses, pages 3, 4, and 8 may be cited
only for narrowly relevant actuator-level claims.

No manual page independently proves a task-level causal diagnosis.

## Robot-reference entry syntax

The review script accepts semicolon-separated references:

    front@412:object outside gripper center;
    wrist@414:jaws close after passing object;
    state@415:tracking remains close to commanded action

Allowed reference channels:

- front
- wrist
- state

`state` creates a robot-sequence reference without a camera field.

## Manual-reference entry syntax

Use:

    4:position feedback specification relevant to tracking premise;
    8:position update rate specification

Do not cite a page without a human-verified relevance statement.

## Freeze boundary

Complete all 90 Pass A cases before creating or opening any Pass B
administrative-reveal artifact.

Pass B must never retroactively rewrite Pass A.
