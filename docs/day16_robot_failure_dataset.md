# Day16 Failure Data Layer — Final Event-Level Contract

## Final correction

Real episode `20260815_111613` contains two distinct operation-anomaly events:

1. `grasp_drop`
2. `post_place_collision`

Therefore one episode-level `failure_interval` / `causal_diagnosis` pair is not
sufficient ground truth.

The final Day16 annotation contract is event-level:

```text
AnomalyReviewCase
└── events[]
    ├── event_id
    ├── observed_failure_mode
    ├── failure_interval
    ├── causal_diagnosis
    ├── supporting_robot_refs
    ├── counterevidence_robot_refs
    ├── confidence
    └── event_status
```

Episode-level fields still preserve:

- `episode_id`
- `task_success`
- `operation_anomaly`
- `original_failure_reason`
- `reviewer`
- diagnostic manifest / frame-record paths.

## Current frozen source truth

The 75-row source audit remains unchanged:

| Category | Count |
| --- | ---: |
| clean reference candidate | 60 |
| operation anomaly | 8 |
| demo quality only | 1 |
| technical exclusion | 6 |
| total | 75 |

The eight operation-anomaly episodes contain **nine events**:

- `20260815_111613`: 2 events
- each of the other seven anomaly episodes: 1 event

The importer must fail if this event count changes unexpectedly.

## Event semantics

`observed_failure_mode` is an observed phenomenon, not a root cause.

Each event begins as:

```text
failure_interval = null
causal_diagnosis = null
supporting_robot_refs = []
counterevidence_robot_refs = []
confidence = null
event_status = draft
```

A verified event requires:

- a smallest defensible interval;
- a causal diagnosis or `insufficient_evidence`;
- confidence;
- supporting robot evidence for any causal diagnosis other than
  `insufficient_evidence`.

## Multi-event rule

Separate physical events must not be merged into one wide interval.

For `20260815_111613`, the grasp-drop event and the post-place-collision event
must receive independent intervals, evidence refs, causal diagnoses, and
confidence values.

## Unchanged components

This correction does not change:

- the 75-row audit taxonomy;
- diagnostic episode binding;
- canonical timestamps;
- state/action semantics;
- the Review Pack 24-frame selection algorithm;
- the eight anomaly episode IDs.

It changes only the human annotation contract from episode-level to event-level.

## Non-claims

Nine draft events are not nine verified diagnoses. Root-cause claims are only
allowed after event-level human review.
