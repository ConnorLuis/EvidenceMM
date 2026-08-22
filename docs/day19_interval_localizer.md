# Day19 Development-Only Interval Proposal Localizer

## Scope

Day19 is the first post-Day18 localization-model selection step.

It converts the frozen Day16/Day17 point candidates into explicit temporal
interval proposals and selects one interval radius using **development Ground
Truth only**.

Day19 does not evaluate the Day18 held-out episodes.

It does not perform physical root-cause classification, Agent orchestration,
MCP planning, robot control, ACT training, or post-held-out tuning.

## Roadmap position

The frozen route is:

```text
Day16
real failure Human GT
    ->
Day17
point-candidate localization diagnostic
    ->
Day18
prospective development / held-out split
    ->
Day19
development-only interval proposal localizer selection
    ->
future stage
held-out interval evaluation
    ->
cross-domain robot + manual evidence diagnosis
    ->
only with real causal supervision:
physical failure-cause classification
```

The project task definition explicitly names temporal boundary error / IoU as a
target metric. Day17 could not report IoU because it emitted isolated points.
Day19 introduces an actual interval-proposal contract, so boundary IoU becomes
semantically valid.

## Inputs

Day19 reuses frozen artifacts:

```text
data/eval/day18_robot_benchmark_split.json
data/annotations/day16_human_gt_events.jsonl
configs/day16_review_pack.yaml
data/manifests/diagnostic_robot_episodes/
data/processed/diagnostic_robot_sequence/
```

The raw robot source remains:

```text
/mnt/f/episodes_pick_place_pilot_v5
```

## Model-selection universe

Day18 freezes:

```text
development:
  54 episodes
  48 clean
   6 anomaly

held_out:
  14 episodes
  12 clean
   2 anomaly
```

Day19 is an **interval localizer within source-reported anomaly episodes**.
Episode-level anomaly detection is not introduced here.

Therefore model selection processes only the six development anomaly episodes.

The development Human GT contains:

```text
7 reviewed events
5 verified intervals
2 reviewed_unresolved
```

Only the five verified intervals participate in supervised boundary metrics.
The two unresolved events remain excluded rather than being treated as
negatives.

## Candidate centers

Day19 does not modify the Day16 selector.

It rebuilds the same frozen pre-GT candidates used by Day17:

```text
uniform anchors
state/action-change peaks
tracking-gap peaks
gripper-action-change peaks
front visual-motion peaks
wrist visual-motion peaks
temporal NMS
```

For interval localization, candidates whose only reason is `uniform_anchor` are
excluded. A frame that is both a uniform anchor and a signal candidate remains
eligible.

Thus interval centers are based on actual signal evidence rather than periodic
coverage anchors.

## Interval proposal

For candidate center `c` and radius `r`:

```text
proposal = [c-r, c+r]
```

clipped to the episode frame range.

No learned neural parameters are introduced. Day19 is a small-data,
development-calibrated interval-proposal baseline.

## Radius model selection

The frozen radius grid is:

```text
0, 1, 2, 3, 4, 5, 6, 8, 10, 12 frames
```

For every radius, Day19 reports development metrics over the five verified
events:

```text
event recall
mean / median best IoU
Recall@IoU 0.10
Recall@IoU 0.25
Recall@IoU 0.50
mean onset absolute error
mean offset absolute error
proposal count
```

The selection objective is lexicographic:

```text
1. maximize event recall
2. maximize Recall@IoU 0.25
3. maximize mean best IoU
4. minimize radius
```

There is no access to held-out metrics during this selection.

## Prospective anti-leakage boundary

Day18 was created after all Day16 reviews already existed, so it is not claimed
to be a pristine never-seen-by-human benchmark.

The correct guarantee is procedural and code-level:

```text
candidate generation:
development anomaly episode IDs only

radius selection:
development verified intervals only

held-out GT used for model selection:
false

held-out metrics reported in Day19:
false
```

The Human-GT loader checks the episode ID before accessing the interval field;
rows outside the development-anomaly allowlist are skipped.

The canonical Day19 model artifact contains aggregate development metrics and
the selected radius, but contains no failure boundaries and no held-out episode
IDs.

## Canonical artifact

Day19 generates and tracks:

```text
data/eval/day19_interval_localizer_model.json
```

This is the frozen model-selection result for the future held-out evaluation.

The detailed development report is runtime output only:

```text
reports/day19_interval_localizer_development.json
```

and should remain ignored by Git.

## Run

```bash
python scripts/select_day19_interval_localizer.py
```

## Validate

```bash
python scripts/validate_day19_interval_localizer.py
```

The validator deterministically rebuilds the development candidate sets,
re-runs the complete radius grid, checks the selected model artifact, verifies
the development report, and rejects any held-out episode ID in either artifact.

## Tests

```bash
pytest -q
```

## Acceptance

Day19 is complete when:

```text
pytest: all pass
selection script: exit 0
validator: valid=true
held_out_gt_used_for_model_selection=false
held_out_metrics_reported=false
git diff --check: clean
```

There is deliberately no target value for development IoU. The selected result
must be preserved honestly rather than tuned to an externally desired number.

## What Day19 does not prove

Day19 does not prove generalization.

Development metrics are model-selection diagnostics only.

The selected radius becomes frozen after Day19. A later stage must evaluate it
unchanged on the two held-out anomaly episodes. Only that later held-out result
may be described as prospective held-out localization performance.
