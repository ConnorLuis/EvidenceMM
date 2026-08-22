# Day20 Frozen-Model Held-Out Interval Evaluation

## Scope

Day20 performs the one frozen-model evaluation that Day19 explicitly deferred.

The evaluated object is exactly the Day19 interval-proposal localizer:

```text
candidate centers:
frozen Day16 signal candidates
excluding uniform-only anchors

interval rule:
[c - r, c + r]

frozen radius:
r = 5 frames
```

Day20 does not search another radius, does not change the candidate selector,
and does not select a new model from held-out metrics.

## Roadmap confirmation

The project task definition requires temporal event localization and boundary
error / IoU as part of the real-robot flagship evaluation.

The frozen route is:

```text
Day16
real failure Human GT
    ->
Day17
point-candidate diagnostic baseline
    ->
Day18
prospective development / held-out episode split
    ->
Day19
development-only interval radius selection
    ->
Day20
frozen-model held-out interval evaluation
    ->
next:
cross-domain robot + manual evidence diagnosis
```

Physical root-cause classification remains deferred because the current verified
events do not contain discriminative causal supervision.

## Held-out universe

Day18 froze:

```text
14 held-out episodes
12 clean-reference episodes
 2 source-reported anomaly episodes
```

Day20 evaluates interval localization only within those two held-out anomaly
episodes:

```text
20260815_112859
20260815_155139
```

Their held-out GT contains two verified events and no reviewed-unresolved event.

The 12 held-out clean episodes are not used because Day19/Day20 are temporal
localizers conditioned on a source-reported anomaly episode; they are not
episode-level anomaly detectors.

## Frozen model contract

The canonical model is:

```text
data/eval/day19_interval_localizer_model.json
```

Day20 pins the Day19 commit and Git blob and verifies:

```text
model schema:
evidencemm_day19_interval_localizer_model_v1

model status:
development_selected_interval_proposal_localizer

model-selection split:
development

selected radius:
5 frames

held-out GT used during Day19 selection:
false

held-out metrics reported during Day19:
false
```

It also rechecks the Day19 provenance hashes for:

```text
Day18 benchmark split
Day16 frozen selector config
Day16 Human GT
```

Any drift fails closed.

## Evaluation ordering

Day20 enforces the following order:

```text
1. load frozen split
2. load and verify frozen Day19 model
3. generate frozen candidates for the two held-out anomaly episodes
4. build proposals using radius = 5
5. only then load held-out failure intervals
6. calculate held-out metrics
7. write the canonical final-evaluation artifact
```

The held-out GT loader skips every non-held-out row before accessing its
`failure_interval` field.

## Metrics

Day20 reports:

```text
Event Recall
Mean / Median Best IoU
Recall@IoU 0.10
Recall@IoU 0.25
Recall@IoU 0.50
Mean onset absolute error
Mean offset absolute error
Per observed-failure-mode metrics
```

Unlike Day17, Day20 has interval predictions, so IoU is semantically valid.

## Correct interpretation of "prospective"

Day18 was created after the complete Day16 Human Review already existed.

Therefore Day20 is **not** claimed to be a pristine never-seen-by-human external
benchmark.

The valid claim is narrower and procedural:

> The episode split was frozen before Day19 model selection. Day19 used only
> development intervals to select radius=5. Day20 then evaluated that unchanged
> model on the held-out anomaly episodes without post-held-out model selection.

The report status is therefore:

```text
prospective_procedural_heldout_interval_evaluation
```

## Evaluation seal

Once Day20 is run, the held-out metrics are known.

Therefore:

```text
same held-out set:
may be reused for diagnostics
may NOT be reused for future model selection
```

If a later localizer is changed using knowledge from Day20 results, a new
prospective claim requires new held-out robot data or a newly frozen untouched
split.

This rule prevents iterative tuning against the final test set.

## Canonical artifact

Day20 writes and tracks:

```text
data/eval/day20_heldout_interval_eval.json
```

This artifact contains the final frozen-model held-out proposals, event-level
results, aggregate metrics, provenance hashes, and evaluation seal.

There is no Day20 radius-selection artifact because model selection is forbidden
at this stage.

## Run

```bash
python scripts/eval_day20_heldout_interval_localizer.py
```

## Validate

```bash
python scripts/validate_day20_heldout_interval_eval.py
```

The validator independently rebuilds the two held-out proposal sets with the
frozen model and verifies byte-semantic equality with the tracked Day20 report.

## Tests

```bash
pytest -q
```

## Acceptance

Day20 is complete when:

```text
pytest: all pass
held-out evaluation: exit 0
validator: valid=true

model_selection_performed=false
radius_tuned_on_held_out=false
post_heldout_tuning_allowed=false

selected_radius_frames=5

git diff --check: clean
```

There is deliberately no minimum held-out score threshold.

The measured held-out result must be preserved whether strong or weak.

## Non-goals

Day20 does not add:

- a new candidate selector;
- a new interval radius;
- an episode anomaly detector;
- physical robot root-cause classification;
- manual-page causal labels;
- Agent / LangGraph orchestration;
- MCP;
- ACT training;
- robot control.

The next flagship step should move from temporal localization toward the
cross-domain evidence contract: localized robot evidence plus relevant manual
evidence plus explicit uncertainty / abstention.
