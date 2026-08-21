# EvidenceMM Day 8 - Visual-Motion-Aware Temporal Evidence

## Goal

Day 8 tests one isolated hypothesis:

> Under the same temporal windows and evidence budget as Day 7, can a
> deterministic image-motion selector provide better temporal evidence than
> uniform midpoint selection without using robot state or action signals?

Day 8 does not use `q_t`, action, gripper state, joint velocity, tracking
error, VLM reasoning, ColQwen, LangGraph, FastAPI, MCP, or robot control.

## Frozen comparison protocol

Day 7 remains unchanged: 30 non-overlapping 2 s timestamp windows, one shared
front/wrist sample per window, 30 shared sample indices / 60 camera images
total, and three frozen human temporal gold events.

Day 8 keeps the same windows and evidence budget. Only the within-window
selection rule changes:

```text
Day 7: sample nearest timestamp midpoint
Day 8: sample with highest deterministic visual-motion score
```

Gate A does not read the Day 7 gold.

## Frozen visual-motion rule

For each camera and sample `t`:

1. decode the original JPEG;
2. apply the camera transform stored in the episode manifest;
3. convert to grayscale;
4. resize to 160 x 120 using Pillow bilinear resampling;
5. compute mean absolute pixel difference against sample `t-1`.

For the first sample of a later window, the predecessor from the previous
window is used. The first sample of the episode receives score 0.

Camera fusion is frozen as:

```text
fused_motion(t) = max(front_motion(t), wrist_motion(t))
```

The selected sample is `argmax fused_motion(t)`. Exact ties choose the lower
frame index. There is no threshold, event-specific weight, or special handling
of the Day 7 lift interval.

## Evidence identity

Motion scoring decodes images but creates no canonical derived image. Selected
evidence keeps the original image path, SHA256, camera, frame index, sample
timestamp, source timestamp, and source age. Source JPEG SHA256 is verified
before scoring.

## Gate A

Gate A validates orientation, grayscale/resize preprocessing, deterministic
mean-absolute-difference scoring, max fusion, lower-index tie breaking, and one
real-window smoke run with `gold_read=false`.

Only after Gate A is accepted will Gate B read the frozen Day 7 gold and
compare event coverage and nearest-evidence temporal distance.

## Gate B

After Gate A freezes the motion rule, Gate B is the first stage allowed to read
the Day 7 temporal gold. It scores all 30 frozen Day 7 windows, selects exactly
one shared front/wrist sample per window, and compares the result directly
against the uniform midpoint baseline.

Gate B reports event coverage, per-event closest-evidence distance to the event
center, mean closest-evidence distance, the complete 30-frame selection
distribution, and the fixed 30-shared-sample / 60-image evidence budget.

No motion preprocessing, fusion, threshold, window size, or gold interval may
be changed after this evaluation.

## Observed Day 8 result

The frozen visual-motion selector was evaluated on the same Day 7 episode,
the same 30 two-second windows, the same three verified temporal events, and
the same evidence budget of 30 shared samples / 60 camera images.

### Aggregate comparison

| Metric | Uniform midpoint | Visual motion | Delta |
| --- | ---: | ---: | ---: |
| Event coverage | 0.6667 | 0.6667 | 0.0000 |
| Covered events | 2 / 3 | 2 / 3 | 0 |
| Mean closest-evidence distance | 344.467 ms | 544.470 ms | +200.003 ms |

The visual-motion selector therefore does **not** improve the Day 7 temporal
baseline on this one-episode smoke evaluation.

### Per-event comparison

| Event | Midpoint | Motion | Midpoint distance | Motion distance |
| --- | --- | --- | ---: | ---: |
| `object_lift` | MISS | MISS | 332.337 ms | 467.676 ms |
| `object_transport` | HIT | HIT | 435.247 ms | 33.338 ms |
| `object_place` | HIT | HIT | 265.817 ms | 1132.397 ms |

The short `object_lift` event remains missed. The closest motion-selected frame
is frame 403, while the frozen midpoint baseline selects frame 405. The
verified lift interval remains 408-412 and is not widened.

The transport event benefits substantially from motion-aware selection, but the
placement event becomes temporally less representative even though it remains
covered.

### Interpretation

Across the evaluated windows, the fused score is usually dominated by the
wrist-camera motion score. This is consistent with a simple pixel-difference
selector being sensitive to wrist-camera ego-motion and other global visual
change, not only task-semantic manipulation events.

This observation is an interpretation of the measured score distribution, not
a separate causal benchmark.

### Frozen negative result

No preprocessing size, camera fusion, threshold, temporal window, or event
interval is retuned after observing this result.

Day 8 is therefore retained as a negative but informative baseline:

```text
uniform midpoint
    -> 2/3 event coverage

visual pixel motion
    -> 2/3 event coverage
    -> worse mean temporal proximity
    -> transport improves, lift still missed
```

The result motivates an independent robot-state/action-aware selector rather
than further tuning this visual-motion baseline against the current gold.
