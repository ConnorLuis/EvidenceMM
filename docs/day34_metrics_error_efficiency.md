# Day34 Metrics, Error Analysis, Latency/GPU and E2E

Day34 is post-evaluation analysis only. Day33 held-out predictions and the Day32 frozen
configuration are immutable. No prompt, calibration, retrieval, model-selection or
result-conditioned regeneration is permitted.

## Static final-result analysis

Day34 recomputes the frozen Day33 metrics from the frozen predictions and GT, then produces:

- five-way confusion matrix and per-class precision/recall/F1;
- pair-group error breakdown;
- parse-failure breakdown;
- raw-score-top to calibrated-final transition analysis;
- clean-control failures;
- gripper-class collapse analysis;
- Day32 development to Day33 held-out generalization gap;
- one per-case post-hoc analysis record for each held-out episode.

These artifacts are descriptive. They do not feed back into the model.

## Efficiency

Day31, Day32 and Day33 prediction logs already contain `latency_sec` and
`peak_gpu_memory_mb`. Their latency is **generation-only**, not full end-to-end latency.
Day34 summarizes those immutable measurements without rerunning held-out inference.

The Day22 roadmap also requests end-to-end latency. Because the single Day33 held-out
evaluation has already been consumed, Day34 must not rerun held-out episodes for timing.
Instead, it performs a label-blind **development-only** profile on five episodes selected by:

`sha256("evidencemm-day34-e2e-profile-v1|episode_id")`

The profile reuses the exact frozen Day32 model/prompt/calibrator and each selected
development episode's exact frozen Day31 12-frame evidence. It never opens GT.

Measured stages:

- evidence preparation;
- processor/input preparation;
- generation;
- parse + calibration;
- warm per-episode E2E from raw episode to calibrated decision;
- one-time model/processor load separately;
- peak allocated CUDA memory.

The profile is operationally resumable. Partial runtime checkpoints are written only below
`reports/day34_e2e_profile_work/`.

## Evidence density

The root-cause diagnostic input uses 12 selected frames from each 900-frame episode:

- evidence density: 1.333...%;
- frame reduction: 98.666...%;
- frame-count compression: 75x.

## Metric-family coverage

The root-cause-v2 Day31–Day33 predictor measures diagnosis and efficiency. It does not
emit a temporal failure interval, and retrieval/manual corpus are disabled, so temporal
localization and grounding are not silently invented in the Day34 final root-cause report.
The earlier Day20 interval benchmark remains a separate historical temporal experiment and
is not merged into Day33 final accuracy.
