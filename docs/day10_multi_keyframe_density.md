# EvidenceMM Day 10 - Multi-Keyframe Temporal Evidence Density Baseline

## Goal

Day 10 is the final temporal micro-baseline in the current EvidenceMM
development phase.

It tests one representation question:

> After three different one-sample-per-window selectors fail to recover the
> short lift event, does increasing timestamp-only temporal evidence density
> improve event coverage without changing the frozen temporal gold?

Day 10 does not introduce a fourth salience score.

## Diagnostic status

The Day 7 temporal gold is already known from previous experiments. Day 10 is
therefore a **diagnostic ablation**, not a blind benchmark.

Anti-tuning discipline is enforced by freezing the complete budget curve and
selection rule before evaluation and reporting every K value rather than only
the best-performing value.

## Frozen source and windows

Day 10 reuses:

- episode `20260815_110415`;
- canonical timestamp source `samples_csv.elapsed_ns`;
- the exact 30 frozen Day 7 two-second `TemporalSlice` objects;
- the same three frozen human temporal events during Gate B evaluation.

No gold interval is changed.

## Frozen budget curve

The only experimental variable is the number of shared evidence samples per
two-second window:

| K | Shared samples across 30 windows | Front/wrist images |
| ---: | ---: | ---: |
| 1 | 30 | 60 |
| 2 | 60 | 120 |
| 3 | 90 | 180 |

No K larger than 3 is evaluated in this diagnostic.

K=1 is required to reproduce the frozen Day 7 midpoint baseline exactly.

## Frozen selection rule

For a window `[start, end)` and `K` evidence samples, target fractions are:

```text
r_j = j / (K + 1)
j = 1, ..., K
```

Target timestamps are:

```text
target_j
=
start
+
r_j * (end - start)
```

Each target maps to the real sample in the frozen window whose canonical
timestamp is nearest to the target. Exact distance ties choose the lower frame
index.

Therefore:

```text
K=1 -> 1/2
K=2 -> 1/3, 2/3
K=3 -> 1/4, 1/2, 3/4
```

The implementation rejects duplicate selected frame indices instead of adding
a post-hoc fallback rule. The canonical episode has sufficient sample density
for K in {1,2,3}.

## Signals prohibited from selection

Day 10 selection is timestamp-only.

It does not use:

- visual pixel motion;
- `q_t`;
- action;
- tracking error;
- gripper-specific weighting;
- VLM reasoning;
- ColQwen;
- optical flow;
- any learned model.

The selected frame indices still reference the original paired front/wrist
JPEG evidence and original SHA256 values.

## Frozen metrics for Gate B

For every K in `{1,2,3}`, report:

1. Event Coverage: a verified event is covered when any selected shared frame
   lies inside its inclusive frozen frame interval.
2. Nearest Evidence Distance: minimum absolute distance from any selected
   sample timestamp to the event-center timestamp.
3. Shared-sample budget.
4. Paired-camera image budget.
5. Marginal change from K=1 to K=2 and from K=2 to K=3.

Every K is reported. The current three-event diagnostic is not used to choose
a production K.

## Gate A

Gate A validates the frozen temporal quantile geometry without reading temporal
gold:

- exact K=1/K=2/K=3 target fractions;
- nearest-real-sample mapping;
- lower-frame-index tie breaking;
- K=1 exact compatibility with the frozen Day 7 midpoint;
- unique selected frames;
- original paired-camera evidence identity;
- deterministic one-window smoke output with `gold_read=false`.

Only after Gate A passes may Gate B run the already-known frozen-gold
diagnostic.

## Scope boundary

After Day 10, EvidenceMM stops expanding temporal micro-baselines and returns
to the main multimodal RAG path: unified document + robot evidence, temporal
citation, video-source adaptation, failure episodes, cross-domain diagnosis,
benchmark expansion, API, Docker/CI and performance evaluation.

## Gate B

Gate B is the evaluation stage for the already-known frozen temporal gold.
It is explicitly a diagnostic ablation rather than a blind benchmark.

For every frozen Day 7 two-second window, Gate B materializes all three
pre-registered temporal-density conditions:

```text
K=1 -> 30 shared samples / 60 paired-camera images
K=2 -> 60 shared samples / 120 paired-camera images
K=3 -> 90 shared samples / 180 paired-camera images
```

Gate B verifies that K=1 reproduces the Day 7 midpoint frame, timestamp,
event coverage and event-center distance exactly.

For each K it reports:

- verified and covered events;
- Event Coverage;
- mean closest-evidence distance to event centers;
- shared-sample budget;
- paired-camera image budget;
- complete selected-frame distribution.

It additionally reports the marginal changes from K=1 to K=2 and from K=2 to
K=3, including additional evidence cost, coverage change and mean-distance
change.

Every K is reported. `choose_best_k=false` remains enforced, and Gate B does
not change quantile locations, temporal windows, duplicate policy, frozen gold,
or any Day 7-9 selector.

## Observed Day 10 result

Gate B evaluated the frozen multi-keyframe density curve on the same episode,
the same 30 Day 7 two-second windows, and the same three frozen verified
temporal events.

This evaluation is a diagnostic ablation with known frozen gold, not a blind
benchmark.

### Aggregate density curve

| K | Shared samples | Paired-camera images | Event coverage | Mean closest-evidence distance |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 60 | 0.6667 (2/3) | 344.467 ms |
| 2 | 60 | 120 | 1.0000 (3/3) | 55.944 ms |
| 3 | 90 | 180 | 1.0000 (3/3) | 122.864 ms |

K=1 exactly reproduces the frozen Day 7 midpoint baseline.

From K=1 to K=2, the additional 30 shared samples / 60 images recover the
previously missed `object_lift` event and reduce mean closest-evidence distance
by 288.523 ms.

From K=2 to K=3, another 30 shared samples / 60 images produce no additional
event-coverage gain and increase mean closest-evidence distance by 66.920 ms on
this three-event diagnostic.

### Per-event result

| Event | K=1 | K=2 | K=3 |
| --- | ---: | ---: | ---: |
| `object_lift` coverage | MISS | HIT | HIT |
| closest frame | 405 | 410 | 412 |
| distance | 332.337 ms | 0.100 ms | 134.241 ms |
| `object_transport` coverage | HIT | HIT | HIT |
| closest frame | 465 | 470 | 472 |
| distance | 435.247 ms | 99.976 ms | 33.338 ms |
| `object_place` coverage | HIT | HIT | HIT |
| closest frame | 645 | 650 | 652 |
| distance | 265.817 ms | 67.757 ms | 201.014 ms |

### Interpretation

The Day 7-9 experiments changed the one-sample selection signal while keeping
one representative sample per two-second window. All three methods retained
2/3 event coverage and missed the short lift event.

Day 10 changes the representation density instead of the salience score.
K=2 recovers the short lift interval with frame 410 while preserving the frozen
timestamp-only quantile protocol. This is evidence that the one-sample
compression itself was an important bottleneck in this episode.

The result does **not** establish K=2 as the production setting. The current
evaluation contains one episode and three already-known frozen events. K=2
happens to dominate K=3 on aggregate distance in this diagnostic, but
`choose_best_k=false` remains in force. A production evidence budget must be
selected only after multi-episode RobotOps evaluation.

### Frozen conclusion

No temporal gold, quantile location, K set, tie-break, duplicate policy, or
Day 7-9 selector is changed after observing the result.

The current temporal evidence ladder is frozen as:

```text
Day 7 midpoint
    30 shared samples / 60 images
    coverage 2/3
    mean distance 344.467 ms

Day 8 visual motion
    30 shared samples / 60 images
    coverage 2/3
    mean distance 544.470 ms

Day 9 state/action change
    30 shared samples / 60 images
    coverage 2/3
    mean distance 699.200 ms

Day 10 timestamp-only density
    K=1: 30 / 60   -> 2/3, 344.467 ms
    K=2: 60 / 120  -> 3/3, 55.944 ms
    K=3: 90 / 180  -> 3/3, 122.864 ms
```

Day 10 is the final temporal micro-baseline in this development phase.
Subsequent work returns to the EvidenceMM system path: unified document + robot
evidence, temporal citations, source adapters for uploaded operation video,
failure episodes, cross-domain diagnosis, benchmark expansion, API, Docker/CI,
and performance evaluation.
