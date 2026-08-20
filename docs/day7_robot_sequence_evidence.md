# EvidenceMM Day 7 - SO-ARM101 Sequence Evidence

## Canonical source

Day 7 uses the original `metadata.json`, `samples.csv`, and front/wrist JPEG
sequences as canonical robot evidence. MP4 is only a future derived display
artifact.

The two camera frames are sample-synchronized, not guaranteed simultaneous
sensor exposures. `samples.csv:elapsed_ns` is the canonical relative timestamp
for each sample pair, while each camera source timestamp and age are preserved.

Gate A builds a tracked EpisodeManifest and an ignored 1800-row FrameRecord
index without copying or re-encoding images. The episode SHA256 is computed
from metadata SHA256, samples CSV SHA256, and all frame SHA256 values in
deterministic `(camera, frame_index)` order.

Joint state and action columns remain in the hashed source CSV but are
deliberately excluded from FrameRecord for the uniform visual baseline.

## Gate B - Timestamp-based temporal slicing

Gate B slices the sample-synchronized episode using the real relative sample
timestamps stored in `samples.csv:elapsed_ns`.

The baseline uses non-overlapping two-second windows:

- `[0, 2)`
- `[2, 4)`
- `[4, 6)`
- and so on.

A window is defined in timestamp space, not by assuming a fixed number of
frames. Within each window, the shared sample whose timestamp is closest to the
window midpoint is selected. Ties are resolved by the lower frame index.

The same sample frame index is then used for both cameras.

No image is decoded, copied, resized, or re-encoded during midpoint selection.
The slice simply reuses the original front/wrist `image_relpath` and SHA256
already bound by Gate A.

Each `TemporalSlice` records the slice group, start/end timestamp, frame-index
range, midpoint target, selected real sample timestamp, midpoint temporal
error, shared midpoint frame index, and the original front/wrist evidence
identity.

The camera-specific source timestamps and ages remain attached to each
midpoint evidence record. Therefore a pair shares a robot sample timestamp
without falsely claiming simultaneous sensor exposure.

## Gate C - Human temporal event gold

Gate C evaluates whether the frozen uniform two-second midpoint baseline
actually observes short, human-defined manipulation events.

Human annotation is performed on the original front/wrist image sequence, not
on the midpoint outputs. This avoids labeling bias toward the baseline being
evaluated.

The initial candidate set considered `approach_object`, `grasp_close`, and
`object_lift`. Human review of the original paired image sequence showed that
`approach_object` is left-censored at episode start and the exact
`grasp_close` boundary is visually ambiguous / partially occluded. Those two
events are therefore excluded rather than assigned manufactured boundaries.

The final pure-visual benchmark uses `object_lift`, `object_transport`, and
`object_place`. For each retained event, a human annotator records the
inclusive start/end frame index using the original numbered JPEG sequence and
changes the annotation status from `draft` to `verified`.

The evaluator resolves those frame bounds back to the canonical
`samples.csv:elapsed_ns` timeline and reports:

- verified event count;
- covered event count;
- event coverage;
- which uniform midpoint slices hit each event;
- closest midpoint to the event center.

A baseline hit occurs only when the selected midpoint frame index lies inside
the verified human event interval.

Misses are retained as real baseline failures. Event intervals must not be
expanded merely to make the uniform midpoint baseline pass.

## Gate C final protocol and smoke result

For episode `20260815_110415`, the verified high-confidence intervals are:

| Event | Inclusive frame interval | Duration |
| --- | ---: | ---: |
| `object_lift` | 408-412 | 0.268 s |
| `object_transport` | 413-530 | 7.802 s |
| `object_place` | 630-668 | 2.534 s |

Frames 531-629 are intentionally left as an alignment / descent transition
gap rather than being forced into either transport or placement. The exact
gripper-release completion boundary is also not used as a separate visual gold
event because it cannot be localized reliably from the paired images alone.

The frozen two-second uniform-midpoint baseline gives:

- verified events: 3;
- covered events: 2;
- event coverage: `2/3 = 0.6667`;
- `object_lift`: missed; nearest midpoint is frame 405, 332.337 ms from the
  annotated event center;
- `object_transport`: covered by midpoint frames 435, 465, 495, and 525;
- `object_place`: covered by midpoint frame 645.

This is a one-episode temporal smoke baseline, not a headline benchmark.
The miss on the 0.268-second lift event is retained as a real baseline failure.
It motivates later comparison against visual-motion-aware and robot-state /
action-aware temporal evidence selection. The verified intervals must not be
expanded and the two-second window must not be retuned merely to remove this
failure.
