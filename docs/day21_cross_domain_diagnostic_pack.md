# Day21 Cross-Domain Diagnostic Evidence Pack

## Scope

Day21 begins the post-localization cross-domain stage required by the EvidenceMM
flagship route.

It does **not** yet claim automated physical root-cause diagnosis.

Instead it connects:

```text
Day20 localized failure proposal
+
canonical robot front/wrist evidence
+
robot observation/action/tracking_error
+
canonical Day15 manual retrieval
+
explicit evidence-sufficiency / abstention policy
```

into a validated `UnifiedEvidenceBundle` for each frozen diagnostic case.

## Why Day21 is the next roadmap step

The project task definition requires the final robot-failure path to combine:

- localized robot evidence;
- visible cameras;
- state/action evidence;
- supporting manual evidence;
- uncertainty / unanswerable behavior.

Day20 completed the frozen-model temporal held-out evaluation and explicitly
states that the next flagship step should move toward:

```text
localized robot evidence
+
relevant manual evidence
+
explicit uncertainty / abstention
```

Day21 implements that evidence bridge before any VLM is allowed to emit a
physical-cause answer.

## Important diagnostic-analysis boundary

The two Day20 held-out results are already exposed.

Day20 explicitly allows the same held-out cases to be reused for diagnostics,
but not for future model selection.

For Day21, each case starts from the Day20 `best_proposal_*` that was matched to
the Human-GT interval during final evaluation.

Therefore the localization origin is deliberately named:

```text
day20_gt_matched_best_proposal_for_post_eval_diagnostics
```

This is suitable for post-evaluation diagnostic evidence analysis.

It is **not** an end-to-end inference claim.

Day21 never copies these Day20 review-label fields into its canonical artifact:

```text
gold_start_frame
gold_end_frame
observed_failure_mode
best_iou
onset_abs_error_frames
offset_abs_error_frames
```

## Robot evidence

For each matched Day20 proposal:

```text
[start_frame, center_frame, end_frame]
```

Day21 binds three representative robot samples.

Each sample contains:

```text
front image
wrist image
canonical timestamp
observation
action
tracking_error
```

and uses the existing `UnifiedEvidenceItem` / `EvidenceRef` contract.

Every selected image is SHA256-verified against the frozen episode/frame
manifests.

## Manual retrieval

Day21 reuses the canonical Day15 document stack:

```text
BM25 Top-5
       \
        -> union -> BGE reranker -> Top-3
       /
BGE-M3 Top-5
```

The manual source remains the currently indexed:

```text
STS3215 datasheet
```

The retrieval query is fixed and failure-label-independent:

```text
STS3215 servo operating limits torque load overload stall feedback
position voltage protection robot gripper operation troubleshooting
```

It does not include the reviewed labels such as:

```text
grasp_drop
post_place_collision
target_offset_or_perception
gripper_close_timing
trajectory_execution_deviation
```

## Why retrieved manual pages are not causal labels

The currently indexed STS3215 document is a component datasheet.

Retrieval rank only means:

> these are the highest-ranked available manual pages for the generic diagnostic
> query.

It does **not** mean:

> this page proves the physical root cause of the robot failure.

There is currently no human-verified mapping from a robot failure event to a
manual page that establishes causal support.

Therefore Day21 records:

```text
manual_support_status:
retrieved_candidates_unlabeled_for_causal_support
```

## Explicit abstention contract

Each case receives a deterministic readiness result.

Even when the cross-domain bundle is structurally valid, Day21 sets:

```text
root_cause_answerable = false
decision = abstain_physical_root_cause
```

because:

1. the Day20 proposal used here is GT-matched for post-evaluation diagnostics,
   not an end-to-end event-selection output;
2. retrieved manual pages are candidate evidence, not verified causal support;
3. the current real-event annotations do not provide discriminative physical
   causal supervision.

This prevents a later generator from turning document proximity into fabricated
physical causality.

## Canonical artifact

Day21 writes:

```text
data/eval/day21_cross_domain_diagnostic_cases.json
```

The artifact contains:

- frozen provenance hashes;
- one canonical manual retrieval trace;
- two diagnostic cases;
- the matched localized proposal for each case;
- three robot evidence samples per case;
- three retrieved document pages per case;
- complete `UnifiedEvidenceBundle` objects;
- explicit readiness / abstention reasons.

## Run

```bash
python scripts/build_day21_cross_domain_diagnostic_pack.py
```

The canonical hybrid retriever loads BGE-M3 and the BGE reranker just as the
frozen Day15 path does.

## Validate

```bash
python scripts/validate_day21_cross_domain_diagnostic_pack.py
```

The validator does not rerun the GPU document ranker.

Instead it verifies:

- frozen Day20 Git blob and provenance;
- exact Day20 case/proposal binding;
- no Day20 review-label fields leaked into Day21;
- document evidence matches the canonical PDF-page builder;
- robot evidence frames match start/center/end of the localized proposal;
- robot timestamps, images, state/action and hashes match canonical raw sources;
- every bundle satisfies the existing cross-domain evidence contract;
- both cases remain `root_cause_answerable=false`.

## Tests

```bash
pytest -q
```

## Acceptance

Day21 is complete when:

```text
pytest: all pass

build pack:
exit 0
case_count = 2
document_items_per_case = 3
robot_items_per_case = 3
root_cause_answerable_count = 0
abstain_count = 2

validator:
valid = true
review_label_fields_leaked = false
physical_root_cause_claimed = false

git diff --check:
clean
```

There is no manual-page quality threshold in Day21 because no manual relevance
Ground Truth exists yet.

Do not tune the document retriever against these two diagnostic cases.

## What Day21 establishes

After Day21, EvidenceMM has a real bridge from temporal localization into the
existing cross-domain evidence contract:

```text
real failed episode
    ->
localized interval
    ->
front/wrist + state/action evidence
    ->
manual retrieval
    ->
UnifiedEvidenceBundle
    ->
evidence-sufficiency gate
```

The next step may run grounded multimodal generation over these frozen bundles,
but that generator must respect the Day21 readiness contract and abstain from a
specific physical root cause until stronger causal evidence exists.

## Non-goals

Day21 does not add:

- a new localizer;
- new held-out model selection;
- a physical root-cause classifier;
- manual causal Ground Truth;
- failure-cause macro-F1;
- Agent / LangGraph;
- MCP;
- ACT training;
- robot control.
