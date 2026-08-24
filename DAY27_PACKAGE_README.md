# EvidenceMM Day27 ambiguity protocol v2 patch

This patch replaces the retired `nominal_attempt_failure_capture` operationalization.
It does **not** change the frozen Day22 taxonomy, 90-row layout, paired-control rule,
anti-label-leakage rule, or Day30 split rule. No canonical s06 was accepted under v1.

## Why v1 is retired

A practiced human operator produced two technically valid nominal task successes for
`rcv2_g01_s06`. Waiting for an accidental natural failure is therefore not a
reproducible collection strategy. Both attempts remain raw noncanonical attempts.

## Frozen v2 ambiguity protocol

Each s06 gets exactly one admin-only single-cause challenge in a 5/5/5 rotation:

- g01/g04/g07/g10/g13: target offset 20 mm Follower-forward (Day23 mild pilot, failure)
- g02/g05/g08/g11/g14: late gripper close after 30-40 mm upward progress (Day23/Day25 frozen failure-inducing setting)
- g03/g06/g09/g12/g15: trajectory deviation about 25 mm Follower-forward (Day23 mild pilot, failure)

The Day22 row remains `planned_physical_cause=unknown` and
`planned_intervention_type=ambiguity_protocol`. The assignment is administrative
metadata and must never be model-visible.

Day27 canonical acceptance means only: technically valid, the assigned single
challenge was applied, no second intervention, comparable scene, and task failure.
It does **not** mean `insufficient_evidence` is already Ground Truth. On Day29, the
human reviewer must judge answerability from model-visible evidence first and only
then reveal the admin challenge assignment.

## Preserve the two v1 attempts

Before applying this patch, register both successful nominal attempts with the
current v1 updater. They must remain noncanonical raw attempts. The patch deliberately
does not overwrite `data/protocol/day27_insufficient_evidence_collection_records.csv`.
