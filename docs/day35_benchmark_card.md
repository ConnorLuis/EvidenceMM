# Day35 Benchmark Card

## Scope

This card describes the final `root-cause-v2` benchmark state frozen through
Day34 and published during Day35 closure.

## Population

- Canonical episodes: 90
- Pair groups: 15
- Development: 60 episodes / 10 pair groups
- Held-out: 30 episodes / 5 pair groups
- Pair-group cross-split count: 0

The held-out ground-truth support in the single final evaluation is:

| Ground truth | Support |
| --- | ---: |
| target_offset_or_perception | 5 |
| gripper_close_timing | 9 |
| trajectory_execution_deviation | 11 |
| insufficient_evidence | 0 |
| clean_success | 5 |

Because the held-out set contains no `insufficient_evidence` GT support,
`false_answer_rate` is not estimable for this split and remains `null`.

## Protocol chronology

The benchmark was designed to make result-conditioned tuning visible.

1. Day22 froze the benchmark protocol and future groupwise split rule.
2. Day23 excluded pilot data from the final benchmark.
3. Day24-Day27 collected the final intervention/control evidence.
4. Day28 bound and audited canonical raw sources.
5. Day29 froze human causal ground truth.
6. Day30 materialized the precommitted groupwise split.
7. Day31 froze the zero-shot baseline on development only.
8. Day32 calibrated on development only.
9. Day33 committed and pushed a one-shot held-out authorization before inference.
10. Day34 performed post-hoc analysis without rerunning held-out episodes.

## Model / evidence contract

Final model:

`Qwen/Qwen3-VL-4B-Instruct`

Evidence per episode:

- 12 selected frames from 900;
- front + wrist contact-sheet imagery;
- observation/action evidence;
- deterministic state/action selection inherited from the Day31 contract.

The Day32 frozen decision transform uses class log-biases:

- target: +1.0
- gripper: -0.5
- trajectory: +1.0
- clean: 0.0

with confidence threshold 0.0 and margin threshold 0.0. Parse failure is
fail-closed to `insufficient_evidence`.

## Frozen held-out result

| Metric | Value |
| --- | ---: |
| Answerable 3-class Macro-F1 | 0.167224 |
| Failed-case 4-way Macro-F1 | 0.125418 |
| Substantive 4-class Macro-F1 | 0.115714 |
| Abstention accuracy | 0.880000 |
| False-abstention rate | 0.120000 |
| Clean false-positive cause rate | 0.600000 |
| Clean accuracy | 0.000000 |
| Overall decision accuracy | 0.166667 |
| Prediction parse rate | 0.833333 |

Final decision counts:

- target: 9
- gripper: 0
- trajectory: 14
- insufficient evidence: 5
- clean: 2

## Per-class held-out report

| Class | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| target | 5 | 9 | 0.1111 | 0.2000 | 0.1429 |
| gripper | 9 | 0 | n/a | 0.0000 | 0.0000 |
| trajectory | 11 | 14 | 0.2857 | 0.3636 | 0.3200 |
| insufficient | 0 | 5 | 0.0000 | n/a | 0.0000 |
| clean | 5 | 2 | 0.0000 | 0.0000 | 0.0000 |

## Main observed failure modes

### Calibration-induced gripper collapse

Before calibration, 14 of 25 parse-success held-out cases had raw
`gripper_close_timing` as the top substantive score. Frozen calibration moved
all 14 away from gripper: 9 to target and 5 to trajectory.

The frozen final classifier predicted gripper zero times. This means the
final zero gripper recall should not be simplified to "the VLM never saw
gripper evidence". The raw scorer frequently ranked gripper first; the
development-fitted transform did not generalize.

### Structured-output failure

Five responses raised `JSONDecodeError`. All five were fail-closed to
`insufficient_evidence`. Their GT labels were 2 clean, 2 gripper and 1
trajectory.

### Clean-control over-diagnosis

Of 5 clean controls:

- 0 were correctly classified clean;
- 3 were classified as physical causes;
- 2 failed closed.

## Generalization gap

| Metric | Development | Held-out | Held-out - Dev |
| --- | ---: | ---: | ---: |
| Answerable 3-class Macro-F1 | 0.318519 | 0.167224 | -0.151294 |
| Failed-case 4-way Macro-F1 | 0.238889 | 0.125418 | -0.113471 |
| Substantive 4-class Macro-F1 | 0.287829 | 0.115714 | -0.172115 |
| Clean accuracy | 0.300000 | 0.000000 | -0.300000 |

The development-to-held-out drop is part of the benchmark result, not something
to be tuned away after the final evaluation.

## Efficiency

Frozen held-out generation-only latency:

- mean 10.040 s
- P50 9.738 s
- P95 13.366 s

Mean peak allocated GPU memory: 9427.9 MB.

The separate Day34 development-only E2E profile reported warm mean 9.194 s,
P95 10.122 s and 6.53 episodes/minute.

## Interpretation

The benchmark demonstrates a complete evidence/diagnostic/evaluation workflow,
not a solved classifier. Its strongest claims are protocol discipline,
traceability, failure analysis and reproducible frozen evaluation.
