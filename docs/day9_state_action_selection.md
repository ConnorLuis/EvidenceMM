# EvidenceMM Day 9 - Robot-State/Action-Aware Temporal Evidence

## Goal

Day 9 tests one isolated hypothesis:

> Under the same episode, frozen two-second windows, verified temporal gold,
> and 30-shared-sample evidence budget, can robot proprioceptive/control
> signals select more useful temporal evidence than uniform midpoint or naive
> visual pixel motion?

Day 9 does not modify the Day 7 midpoint baseline or the frozen Day 8 visual
motion baseline.

## Gate A0 source-semantics audit

The canonical `samples.csv` contains 900 rows and 63 columns. Six robot joints
are represented consistently:

- `shoulder_pan`
- `shoulder_lift`
- `elbow_flex`
- `wrist_flex`
- `wrist_roll`
- `gripper`

For every joint the source exposes:

```text
leader
leader_delta
observation
desired_unclamped
desired_clamped
action
command_lag
tracking_error
```

The source metadata explicitly defines:

```text
observation =
Follower Present_Position read before the current action write

action =
Final absolute Goal_Position actually sent after mapping,
session clamp and rate limiting
```

Day 9 therefore defines:

```text
q_t = observation_* 6D
a_t = action_* 6D
```

The audit also verifies over all 900 rows:

```text
tracking_error = abs(action - observation)
command_lag = abs(desired_clamped - action)
desired_unclamped = initial_desired + leader_delta
```

`leader_delta` is not assumed to equal `leader_t - first_csv_leader` because
that exact invariant does not hold. Leader signals are excluded from the Day 9
primary selector.

Wrist flex is clamped on 342 rows, and action differs from
`desired_clamped` on some rows for several joints, consistent with the source
metadata stating that `action` is the final post-clamp/post-rate-limit command.

## Frozen Gate A rule

Before reading temporal gold, Day 9 freezes the selector as:

```text
state_change(t)  = RMS(q_t - q_(t-1))
action_change(t) = RMS(a_t - a_(t-1))

state_action_score(t)
    = max(state_change(t), action_change(t))
```

All six joints receive equal weight. There is:

- no per-joint weighting;
- no gripper boost;
- no normalization fitted to this episode;
- no threshold;
- no visual input.

The observed no-gold scale audit supports using the two change channels
directly:

```text
q RMS delta:
mean 0.167148
max  0.782223

action RMS delta:
mean 0.191654
max  0.843578
```

Their scales are comparable.

The tracking gap is retained for diagnostics only:

```text
tracking_gap(t) = RMS(a_t - q_t)
```

Its observed scale is much larger:

```text
mean 2.257912
max  4.044711
```

and it measures follower/command mismatch rather than transition magnitude.
It therefore does not participate in the selection score.

## Fair comparison

Day 9 reuses the exact frozen Day 7 `TemporalSlice` objects.

```text
Day 7: one midpoint sample per 2 s window
Day 8: one visual-motion argmax sample per 2 s window
Day 9: one state/action-change argmax sample per 2 s window
```

All three methods use:

- the same episode;
- 30 two-second windows;
- 30 shared sample indices;
- 60 corresponding front/wrist evidence images when materialized;
- the same three verified temporal events during later evaluation.

Gate A does not read temporal gold.

## Evidence semantics

The state/action selector answers **when to inspect evidence**. A selected
sample can later point to:

- the 6D observation vector;
- the 6D action vector;
- adjacent state/action deltas;
- tracking error / tracking-gap diagnostics;
- the original front/wrist JPEGs at the same frame index.

No state/action value replaces the canonical source identity already bound by
the episode manifest and `samples.csv` SHA256.

## Gate A acceptance

Gate A validates:

- source metadata semantics;
- all required 6D observation/action/tracking-error columns;
- canonical elapsed timestamp use;
- adjacent-frame RMS state change;
- adjacent-frame RMS action change;
- max fusion;
- lower-frame-index tie breaking;
- tracking gap as diagnostic only;
- one real frozen window smoke run with `gold_read=false`.

Only after Gate A passes and the rule is frozen may Day 9 evaluate against the
Day 7 temporal gold.

## Gate B

Gate B is the first Day 9 stage allowed to read the frozen Day 7 temporal
gold.

It evaluates all 30 frozen two-second windows and compares three methods under
the same episode, gold and 30-shared-sample / 60-image evidence budget:

```text
uniform midpoint
vs
visual motion
vs
state/action change
```

The Day 9 state/action selection rule remains exactly the Gate A rule. Gate B
does not change joint weighting, normalization, fusion, window size, or any
gold interval.

In addition to event coverage and per-event closest-evidence distance, Gate B
records whether each state/action selection was dominated by state change or
action change. Tracking gap remains diagnostic only and never participates in
the selector.

The canonical `metadata.json` and `samples.csv` hashes are verified against the
frozen episode manifest before evaluation.

## Observed Day 9 result

The frozen state/action selector was evaluated on the same episode, the same
30 two-second windows, the same three verified temporal events, and the same
30-shared-sample / 60-image evidence budget as Days 7 and 8.

### Three-way aggregate comparison

| Metric | Uniform midpoint | Visual motion | State/action |
| --- | ---: | ---: | ---: |
| Event coverage | 0.6667 | 0.6667 | 0.6667 |
| Covered events | 2 / 3 | 2 / 3 | 2 / 3 |
| Mean closest-evidence distance | 344.467 ms | 544.470 ms | 699.200 ms |

Relative to uniform midpoint, the state/action selector changes event coverage
by 0.0000 and worsens mean closest-evidence distance by 354.733 ms.

Relative to visual motion, event coverage again changes by 0.0000 and mean
closest-evidence distance worsens by 154.730 ms.

### Per-event comparison

| Event | Midpoint | Visual motion | State/action |
| --- | ---: | ---: | ---: |
| `object_lift` coverage | MISS | MISS | MISS |
| `object_lift` closest frame | 405 | 403 | 399 |
| `object_lift` distance | 332.337 ms | 467.676 ms | 731.899 ms |
| `object_transport` coverage | HIT | HIT | HIT |
| `object_transport` closest frame | 465 | 472 | 469 |
| `object_transport` distance | 435.247 ms | 33.338 ms | 165.274 ms |
| `object_place` coverage | HIT | HIT | HIT |
| `object_place` closest frame | 645 | 632 | 631 |
| `object_place` distance | 265.817 ms | 1132.397 ms | 1200.428 ms |

The short `object_lift` interval remains missed. The verified interval remains
frames 408-412 and is not modified after evaluation.

### Channel dominance

Across the 30 selected windows:

```text
action-dominated = 27
state-dominated  = 3
tie              = 0
```

The frozen `max(state_change, action_change)` rule is therefore overwhelmingly
driven by changes in the final commanded action on this episode.

This is a measured property of the selected windows. It does not by itself
prove that action changes are intrinsically inferior temporal signals.

### Interpretation

The Day 9 result shows that a large robot-state or command transition is not
equivalent to a task-semantic event boundary. In this episode the final action
changes frequently due to the teleoperation/control trajectory, so selecting
the largest within-window 6D change can prefer strong control transitions that
are temporally distant from the human-verified lift/place event centers.

State/action evidence remains useful as diagnostic context because it exposes
what the follower was doing, what command was sent, and how large the tracking
gap was at a selected sample. However, this simple change-magnitude selector
does not improve temporal localization under the current one-sample-per-window
protocol.

### Frozen negative result

No joint weighting, gripper boost, normalization, fusion rule, threshold,
window size, or temporal gold interval is changed after observing the result.

The Day 7-9 temporal sequence is retained as:

```text
uniform midpoint
    -> 2/3 event coverage
    -> mean distance 344.467 ms

visual pixel motion
    -> 2/3 event coverage
    -> mean distance 544.470 ms

robot state/action change
    -> 2/3 event coverage
    -> mean distance 699.200 ms
    -> 27/30 selections action-dominated
```

The result suggests that the next temporal-evidence experiment should change
the evidence-selection formulation rather than tune Day 8 or Day 9 feature
weights against the current three-event gold.
