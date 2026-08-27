# Day31 Root-Cause Diagnostic Baseline

## Objective

Day31 freezes and measures a **development-only, direct zero-shot multimodal baseline** for the root-cause benchmark.

This is not the old Day14 system-pipeline failure diagnosis. Day31 predicts robot-operation diagnostic decisions from the Day22 taxonomy.

## Frozen population boundary

- Development: 60 episodes from the frozen Day30 split.
- Held-out: 30 episodes, inference forbidden on Day31.
- Held-out labels are not aggregated, inspected for tuning, or used for model selection.
- The baseline inference stage does not open Ground Truth.
- Day30's receipt is hash-verified but not parsed, because it contains administrative held-out aggregate label audits.
- During evaluation, held-out GT JSONL lines are skipped by episode ID before JSON parsing.
- Ground Truth is joined only after all 60 development predictions are complete.

## Baseline

Model: `Qwen/Qwen3-VL-4B-Instruct`

Mode: `direct_zero_shot_multimodal_no_retrieval`

The model receives:

- front camera evidence;
- wrist camera evidence;
- observation/action/tracking summaries;
- the frozen diagnostic taxonomy.

It does not receive:

- pair-group IDs;
- intervention metadata;
- physical-cause GT;
- diagnostic-decision GT;
- human review notes;
- development label priors;
- held-out labels.

No manual retrieval is used. This deliberately makes Day31 a direct VLM lower-bound baseline rather than the final EvidenceMM system.

## Evidence sampling

Each 900-frame development episode contributes 12 deterministic frames:

1. seven uniform chronological anchors;
2. five highest fused state/action-change frames;
3. dynamic frames must remain at least 20 frames from every already selected frame.

The fused score is:

`max(adjacent observation RMS, adjacent action RMS)`

This uses only model-visible robot signals and no GT.

The 12 frames are rendered into three chronological contact sheets. Wrist images are rotated CCW 90 degrees, matching the frozen review convention.

## Frozen diagnostic decisions

- `target_offset_or_perception`
- `gripper_close_timing`
- `trajectory_execution_deviation`
- `insufficient_evidence`
- `clean_success`

The model must return strict JSON. Invalid output fails closed to `insufficient_evidence` with confidence `0.0`; the original raw response and parse error remain in the prediction record.

Prompt contract SHA256:

`faee60d40b710005a265ef7c657a2b19921c8b40c41ddda8c3d69d4916dbd79f`

## Primary development metrics

- answerable three-class macro-F1
- failed-case four-way diagnostic macro-F1
- abstention accuracy
- false answer rate
- false abstention rate
- clean-control false-positive-cause rate

The literal four-way macro-F1 uses the fixed four failed-case labels and zero-division=0. The frozen development GT contains no `insufficient_evidence` support, so this literal metric has a structural ceiling below 1.0. A supported-label failed-case macro-F1 is reported as a secondary interpretability metric.

When there are zero insufficient-evidence GT failures, false-answer rate is stored as `null` with a not-estimable note. Held-out cases are never consulted to repair this denominator.

## Runtime/resume

Runtime work is written below:

`reports/day31_baseline_work/`

This directory is gitignored. Predictions are checkpointed after every episode. Re-running `run` resumes from completed development episodes without touching held-out episodes.

## Frozen outputs

- `data/eval/day31_development_baseline_predictions.jsonl`
- `data/eval/day31_development_baseline_metrics.json`
- `data/protocol/day31_baseline_freeze_receipt.json`

## Day31 boundary

Day31 performs no training, calibration, prompt tuning after results, retrieval tuning, or model selection.

Day32 may calibrate using development data only.

Day33 remains the single frozen held-out evaluation.
