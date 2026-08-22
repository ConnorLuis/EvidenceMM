# Day16 Human Review Pack — Event-Level Annotation

## Unchanged frame selection

The Review Pack still selects at most 24 frames per episode using the same
combination of:

- uniform anchors;
- state/action-change peaks;
- tracking-gap peaks;
- gripper-action changes;
- front visual motion;
- wrist visual motion;
- temporal non-maximum suppression.

No selection weight, top-k, stride, or frame cap changed in the event-contract
revision.

## What changed

The review template is now event-level.

Across the eight anomaly episodes there are nine events because
`20260815_111613` contains both:

```text
grasp_drop
post_place_collision
```

Each event receives its own:

```text
failure_interval
causal_diagnosis
supporting_robot_refs
counterevidence_robot_refs
confidence
event_status
```

## Safe migration

Existing unedited v1 `review_template.json` files are automatically migrated to
the v2 event schema when Review Pack generation is rerun.

A v1 file is considered unedited only when:

- `review_status == draft`
- interval is null
- causal diagnosis is null
- confidence is null
- supporting/counterevidence frame arrays are empty
- reviewer is null

If a legacy template contains human edits, generation refuses to overwrite it.

Already-v2 templates are preserved on rerun.

## Human review

For a multi-event episode, review the same chronological candidate frames but
freeze separate intervals for each event.

The candidate-frame algorithm is only a triage aid. It does not assign event
boundaries or causal labels.
