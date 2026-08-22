# Day16 75-Episode Audit Protocol

## Input

Canonical source pool:

```text
F:\episodes_pick_place_pilot_v5
training_manifest.csv
```

WSL equivalent:

```text
/mnt/f/episodes_pick_place_pilot_v5
```

The CSV encoding is `gb18030`.

## Current source partition

Based on the frozen manifest semantics:

### Clean reference candidates — 60

Rows without a recorded `failure_reason`.

These are candidates, not automatically verified success ground truth.

### Operation anomalies — 8

```text
20260815_111613
20260815_112058
20260815_112633
20260815_112859
20260815_140119
20260815_141416
20260815_141657
20260815_155139
```

All eight have `task_success=True`.

This is why Day16 must not use `task_success=False` as the only failure-data
filter.

### Demo-quality-only — 1

```text
20260815_135518
```

Reason:

```text
夹起放下过快，导致后续暂停太长
```

This is useful for imitation-learning quality auditing but is not currently a
physical operation-anomaly diagnosis case.

### Technical exclusions — 6

```text
20260815_153125
20260815_154459
20260815_155524
20260815_155814
20260815_161422
20260815_161647
```

Reasons include follower power loss and wrist duplicate-ratio failures.

These are excluded from the physical operation-anomaly diagnosis pool.

## Execution order

```text
training_manifest.csv
    ↓
import_day16_training_manifest.py
    ↓
75-row source audit
    ↓
8 draft anomaly review cases
    ↓
bind_day16_diagnostic_anomalies.py
    ↓
8 canonical diagnostic manifests + frame indices
    ↓
validate_day16_failure_data_layer.py --require-bound-anomalies
    ↓
manual evidence review
```

No new robot data should be collected until the eight real anomaly episodes
have been reviewed and a missing diagnostic category is demonstrated.
