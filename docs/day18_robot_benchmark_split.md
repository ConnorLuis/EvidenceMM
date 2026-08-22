# Day18 Prospective Robot Benchmark Split

## Scope

Day18 freezes the **prospective development / held-out episode split** for the
next EvidenceMM robot-diagnostic stages.

It does not train a localizer, does not tune the frozen Day16 selector, and does
not claim physical root-cause classification.

Day17 already measured the frozen pre-GT selector on all seven verified failure
events. Therefore Day18 cannot retroactively convert Day17 into a held-out
benchmark. The Day18 split applies only to future Day19+ model selection and
evaluation.

## Why this is the next roadmap step

The project task definition requires the next flagship dataset/evaluation path
to bind real successful and failed robot episodes and eventually support a
held-out episode split.

By the end of Day17 the project has:

- 60 clean-reference candidates from the Day16 source audit;
- 8 source-reported anomaly episodes;
- 9 reviewed anomaly events;
- 7 verified failure intervals;
- 2 reviewed-unresolved events;
- a real temporal-localization diagnostic baseline.

Before developing a new interval localizer or cross-domain diagnostic model, the
episode split must be frozen so later model choices cannot use final held-out
episodes as tuning data.

## Eligible episode universe

Day18 includes only:

```text
clean_reference_candidate: 60
operation_anomaly:          8
--------------------------------
eligible total:            68
```

It excludes:

```text
demo_quality_only:   1
technical_exclusion: 6
```

The exclusions remain source-audit facts; Day18 does not reinterpret them.

## Split protocol

Membership is stratified by source-audit category.

For each eligible category, episodes are ranked with:

```text
sha256(seed|audit_category|episode_id)
```

using the frozen seed:

```text
evidencemm-day18-v1
```

The held-out budget is fixed before future model development:

```text
clean_reference_candidate: 12 / 60
operation_anomaly:           2 / 8
```

Therefore:

```text
development:
  48 clean
   6 anomaly
  54 total

held_out:
  12 clean
   2 anomaly
  14 total
```

No manual cherry-picking is permitted after the split is generated.

## Anti-leakage rule

Split membership is computed from only:

```text
episode_id
audit_category
```

Human Ground Truth is loaded only after membership has already been assigned.

The Day16 failure interval is never an input to split membership.

The tracked Day18 split artifact intentionally does **not** contain:

```text
failure_interval
start_frame / end_frame
start_sec / end_sec
supporting_robot_refs
counterevidence_robot_refs
```

It may contain event identifiers and review dispositions for provenance and
count validation, but not temporal boundary values.

This enables future Day19+ model setup to load the split manifest without
materializing held-out failure boundaries.

## Important status wording

The correct status is:

```text
prospective_heldout_split_frozen_after_day17_baseline
```

Do not say that Day17 was held-out. It was not.

The correct interpretation is:

> Day17 established an honest all-reviewed-event diagnostic baseline. Day18
> subsequently freezes a prospective episode-level held-out partition for all
> future model development and final evaluation.

## Build

```bash
python scripts/build_day18_robot_benchmark_split.py
```

This generates the canonical tracked artifact:

```text
data/eval/day18_robot_benchmark_split.json
```

## Validate

```bash
python scripts/validate_day18_robot_benchmark_split.py
```

Validation recomputes the split deterministically from the frozen Day16 source
audit and checks:

- 75 source-audit rows remain unchanged in category counts;
- 68 eligible episodes are exactly 60 clean + 8 anomaly;
- the split is exactly 54 development + 14 held-out;
- development / held-out episode IDs do not overlap;
- all 8 anomaly episodes align with the Day16 human-GT episode universe;
- human GT remains 9 events = 7 verified + 2 reviewed-unresolved;
- split membership is reproducible from the fixed seed;
- the tracked split artifact contains no failure boundary fields.

## Tests

```bash
pytest -q
```

## Day18 acceptance

Day18 is complete when:

```text
pytest: all pass
build split: exit 0
validator: valid=true
git diff --check: clean
```

The Day18 result must then be committed and frozen before Day19 changes any
future localization or diagnostic model.

## Next roadmap step

After Day18 is frozen, a future localizer or cross-domain diagnostic component
must use:

```text
development episodes -> model selection / design
held_out episodes     -> final evaluation only
```

Day18 itself performs no post-GT model tuning.
