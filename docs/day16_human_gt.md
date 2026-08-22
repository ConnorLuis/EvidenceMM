# Day16 Human Ground Truth

Day16 human review covers 8 real robot-operation
episodes containing 9 source-reported anomaly events.

The human-review layer deliberately distinguishes
source-reported anomalies from verified Ground Truth.

## Review disposition

Canonical promoted annotations use:

- `verified`
  - a defensible failure interval exists;
  - synchronized robot evidence supports the observed event;
  - a causal diagnosis is recorded;
  - `insufficient_evidence` is valid when the failure event
    is observable but the physical cause cannot be uniquely
    established.

- `reviewed_unresolved`
  - the source manifest reports an anomaly;
  - human evidence review has been completed;
  - the synchronized evidence cannot support a defensible
    failure interval;
  - no interval, causal diagnosis, confidence, or supporting
    robot reference is fabricated.

The underlying Day16 annotation contract remains unchanged.
`event_status=draft` therefore continues to mean that the
event itself has not been promoted to verified Ground Truth.

The canonical human-GT layer adds
`review_disposition=reviewed_unresolved` so a completed
negative review is distinguishable from an event that has
not yet been reviewed.

## Final Day16 review result

- episodes: 8
- anomaly events: 9
- verified Ground Truth events: 7
- reviewed but unresolved events: 2
- waiting for review: 0

Reviewed-unresolved events:

- `20260815_140119_event_01`
  (`object_push_during_grasp`)
- `20260815_141657_event_01`
  (`post_place_collision`)

These events preserve their source-reported anomaly labels,
but no failure interval or causal diagnosis is invented.

Canonical promoted artifacts:

- `data/annotations/day16_human_gt_events.jsonl`
- `data/annotations/day16_human_gt_summary.json`

## Promotion

```bash
python scripts/promote_day16_human_gt.py

```

## Validation

```bash
python scripts/validate_day16_human_gt.py
```
