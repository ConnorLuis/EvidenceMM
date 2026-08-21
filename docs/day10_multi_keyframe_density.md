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
