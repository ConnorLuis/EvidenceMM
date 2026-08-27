# Day35 Limitations and Non-Claims

## Final capability boundary

EvidenceMM demonstrates an auditable multimodal evidence and robot-diagnostic
pipeline. It does not demonstrate that physical root-cause diagnosis is solved.

The single frozen held-out result is deliberately modest:

- answerable 3-class Macro-F1: 0.1672;
- overall decision accuracy: 0.1667;
- clean accuracy: 0.0;
- gripper recall: 0.0.

These values are part of the final release and must not be replaced with
development metrics in project claims.

## Dataset size and composition

The final benchmark has 90 canonical episodes and 15 pair groups. The held-out
set has 30 episodes. This is enough to enforce a real development/held-out
workflow, but it is not a large benchmark and should not support broad
claims about general robot-failure generalization.

The held-out split has no GT `insufficient_evidence` cases, so false-answer rate
cannot be estimated there.

## Model-output reliability

The Day32/Day33 structured scoring prompt has an 83.33% parse rate on both
development and held-out. JSON parsing failures are fail-closed, but they still
cause diagnostic errors.

A future system should consider constrained structured decoding or a
schema-native generation mechanism. That future change must not be retrofitted
into the frozen Day33 result.

## Calibration generalization

The development-fitted log-bias calibration strongly suppresses the gripper
class on held-out data. Raw scoring ranks gripper first 14 times, yet the frozen
final transform predicts gripper zero times.

This is evidence of calibration/distribution-shift risk. It is not appropriate
to tune the bias after seeing held-out outcomes.

## Clean-success discrimination

No held-out clean control is correctly predicted. The system tends to
over-diagnose abnormality or fail closed.

Future work should explicitly model negative/clean evidence instead of treating
clean as merely another competing class score.

## Temporal localization and grounding

The final Day31-Day33 root-cause predictor does not emit a temporal failure
interval. The repository contains a separate historical Day20 temporal
experiment, but those metrics are not merged into the final root-cause result.

The final root-cause baseline also uses no manual retrieval corpus, so grounding
metrics from the earlier RAG branch are not presented as Day33 diagnostic
grounding quality.

## Efficiency scope

Day33 latency is generation-only and hardware-specific.

Day34 E2E profiling is development-only by design, because the one-shot held-out
evaluation had already been consumed. It measures system performance, not
additional accuracy.

## Production limitations

The current project does not provide:

- production API deployment;
- multi-user authorization;
- online robot monitoring;
- real-time closed-loop intervention;
- SLA/SLO guarantees;
- distributed model serving;
- Agent/MCP orchestration.

Those are intentionally outside this project phase.
