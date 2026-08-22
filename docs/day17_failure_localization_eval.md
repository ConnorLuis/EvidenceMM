# Day17 Real Failure Temporal Localization Evaluation

## Scope

Day17 is the first supervised evaluation built on the frozen Day16 human
Ground Truth.

It does **not** add Agent orchestration, MCP, robot control, ACT training, or
physical root-cause classification.

The Day16 data layer contains:

- 8 source-reported anomaly episodes;
- 9 anomaly events;
- 7 verified events with defensible failure intervals;
- 2 reviewed-but-unresolved events with no fabricated interval.

All verified Day16 causal diagnoses are `insufficient_evidence`. Therefore
Day17 does not report failure-cause macro-F1 and does not claim automated
physical root-cause diagnosis.

## Why this is the next roadmap step

The project task definition names real failed episodes, failure intervals,
uncertainty/unanswerable labels, relevant robot evidence, and temporal event
metrics as the next flagship evaluation direction.

Day16 supplied the reviewed real-failure labels. Day17 measures how well the
already-frozen pre-GT candidate selector localizes those verified failure
intervals before any post-label tuning.

This keeps the sequence:

```text
real source audit
    ->
human failure interval GT
    ->
blind candidate localization measurement
    ->
future localization model / cross-domain diagnosis
```

rather than tuning a selector against the same labels before establishing a
baseline.

## Frozen candidate selector

Day17 reuses the Day16 Review Pack candidate selector without modifying it:

- uniform anchors;
- state/action-change peaks;
- tracking-gap peaks;
- gripper-action-change peaks;
- front visual-motion peaks;
- wrist visual-motion peaks;
- temporal NMS;
- maximum 24 selected frames per episode.

The selector configuration is loaded from:

```text
configs/day16_review_pack.yaml
```

The Day17 report records the SHA256 of that frozen config.

Candidate sets for all 8 episodes are generated **before** the human-GT JSONL
is loaded. Gold failure intervals and observed failure modes are not inputs to
candidate selection.

## Metrics

For each of the 7 verified events:

```text
exact hit:
  at least one selected candidate frame lies inside
  [start_frame, end_frame]

tolerance hit:
  minimum candidate-to-interval distance <= N frames

minimum frame distance:
  0 when a candidate lies inside the interval

minimum time distance:
  candidate timestamp distance to the closest interval boundary,
  or 0 inside the interval
```

The configured tolerance diagnostics are:

```text
±5 frames
±15 frames
```

Aggregate reporting includes:

- exact event recall;
- tolerance event recall;
- mean/median minimum frame distance;
- mean/median minimum time distance;
- per-observed-failure-mode metrics.

The 2 `reviewed_unresolved` events are reported but excluded from supervised
interval denominators. They are **not** treated as negatives.

## What Day17 intentionally does not report

### Boundary IoU

The frozen selector outputs isolated candidate points, not predicted temporal
intervals. IoU would therefore be semantically invalid. A later interval
localizer may introduce boundary IoU.

### Failure-cause macro-F1

The verified Day16 events do not contain discriminative physical cause labels;
their causal diagnosis is `insufficient_evidence`. Macro-F1 is deferred until
real cause supervision exists.

### Held-out benchmark claims

The current set is a small real-failure diagnostic from one pilot collection,
not a held-out production benchmark. Day17 establishes an honest baseline and
does not tune a quality threshold after seeing GT.

## Run

```bash
python scripts/eval_day17_failure_localization.py
```

The report is written to:

```text
reports/day17_failure_localization_eval.json
```

Validate integrity with:

```bash
python scripts/validate_day17_failure_localization.py
```

Run regression tests with:

```bash
pytest -q
```

## Acceptance contract

Day17 is structurally valid when:

- all 8 configured anomaly episodes produce a candidate set;
- candidate sets respect the frozen Day16 selector budget;
- candidate timestamps bind exactly to canonical `samples.csv`;
- the human-GT universe remains 9 events = 7 verified + 2 unresolved;
- unresolved events remain outside supervised interval metrics;
- report metrics reproduce deterministically from reported candidates + GT;
- no gold interval is read during candidate generation.

There is deliberately no minimum recall threshold in Day17. The measured
result is a baseline, not a target to tune against.
