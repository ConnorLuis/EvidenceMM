# Day29 Pass B + Ground Truth Freeze Operator Guide

## Scope

This phase starts only after Pass A is frozen at commit
`1ae4556d73c8dd409ee4e25d0025fce8a3064a1a`.

Pass B may now reveal the frozen Day24-Day27 administrative collection evidence.
It must **not** rewrite any Pass A field. It must **not** materialize the Day30 split.

The ground-truth label is derived only from frozen administrative collection evidence.
`blind_cause_hypothesis` is never used to construct or validate `physical_cause_gt`.

## Tooling files

Copy these three files into the repository:

- `scripts/day29_pass_b.py`
- `data/protocol/day29_pass_b_operational_contract.json`
- `docs/day29_pass_b_operator_guide.md`

Commit the tooling before running `build` or `freeze`.

Recommended tooling commit:

```bash
git add \
  scripts/day29_pass_b.py \
  data/protocol/day29_pass_b_operational_contract.json \
  docs/day29_pass_b_operator_guide.md

git commit -m "feat(day29): freeze pass b ground truth tooling"
git push origin master
```

## Commands

```bash
python scripts/day29_pass_b.py preflight
python scripts/day29_pass_b.py build
python scripts/day29_pass_b.py validate
python scripts/day29_pass_b.py freeze
python scripts/day29_pass_b.py audit
```

### `preflight`

Checks:

- Pass A freeze commit is an ancestor.
- Day24-Day28 frozen commits are ancestors.
- every required frozen source has the exact expected Git blob;
- Pass A records SHA256 still equals the frozen receipt;
- Pass A is complete and admin reveal had not started at its freeze;
- Day28 says 90 canonical episodes, no registered technical exclusions,
  exactly 2 experimental exclusions, no GT/answerability/split materialized;
- no Day29 GT output already exists.

### `build`

Creates:

`data/annotations/day29_ground_truth_records.jsonl`

in the **frozen Pass A review order**.

The record fields are:

- `schema_version`
- `review_position`
- `episode_id`
- `pair_group_id`
- `technical_valid`
- `experimental_valid`
- `task_success`
- `intervention_verified`
- `physical_cause_gt`
- `evidence_answerability_gt`
- `diagnostic_decision_gt`
- `confidence`
- `review_notes`

`confidence=1.0` denotes deterministic administrative-label verification,
not a model confidence score.

For clean controls, `intervention_verified=false` means
“no intervention exists by design / not applicable”, not an intervention failure.

### `validate`

Fail-closed validation of all 90 GT records.

Expected final counts:

- 90 technical valid
- 90 experimental valid
- 15 clean successes
- 75 controlled/challenge failures
- physical causes:
  - `none_clean`: 15
  - `target_offset_or_perception`: 25
  - `gripper_close_timing`: 25
  - `trajectory_execution_deviation`: 25
  - `unknown`: 0
- diagnostic decisions:
  - `clean_success`: 15
  - target: 25
  - gripper: 25
  - trajectory: 25
  - `insufficient_evidence`: 0

### `freeze`

Creates:

`data/protocol/day29_ground_truth_freeze_receipt.json`

The receipt records source SHA256 values, record SHA256, counts, and the
tooling commit. It sets:

- `human_review_completed=true`
- `admin_reveal_started=true`
- `ground_truth_frozen=true`
- `future_split_materialized=false`

### `audit`

Re-validates both records and freeze receipt.

## Final Git closure

After `freeze` and `audit`:

```bash
pytest -q

git add \
  data/annotations/day29_ground_truth_records.jsonl \
  data/protocol/day29_ground_truth_freeze_receipt.json

git diff --cached --check
git diff --cached --name-only

git commit -m "feat(day29): freeze root cause ground truth"
git push origin master
```

The exact final staged set must contain only those two output files.

After push:

```bash
python scripts/day29_pass_b.py audit
git status --short
```

A clean worktree plus `DAY29 PASS B AUDIT: PASS` closes Day29.

## Day29 boundary after closure

Allowed next action: Day30 pair-group split materialization.

Forbidden during Day29:

- modifying Pass A blind judgments;
- tuning against future held-out data;
- materializing Day30 split early;
- training/calibrating the diagnostic model.
