# Day32 Development-Only Calibration

Day31 froze a valid baseline whose 60 development predictions all became `insufficient_evidence`.
Day32 keeps Day31 immutable and separates **substantive scoring** from **abstention calibration**.

The same Qwen3-VL model and the exact Day31 12-frame evidence are reused. Each Day31 evidence
fingerprint is reconstructed from raw files before Day32 scoring. The scoring prompt is
non-abstaining and returns support scores for three physical causes plus `clean_success`.

The ten development pair groups are deterministically divided into 6 calibration-fit groups
and 4 reporting-only internal-validation groups with seed
`evidencemm-day32-internal-calibration-v1`. Calibration parameters are selected only on the
six fit groups. Validation labels never change the selected parameters, and there is no refit.

The frozen grid contains 3750 combinations of three cause log-biases, confidence threshold,
and top-vs-second margin threshold. Clean bias is fixed to zero. The fit objective is balanced
substantive four-class macro-F1, followed by frozen deterministic tie-breaks.

Day32 runs **zero held-out inferences**, parses **zero held-out GT label rows**, and does not
parse the Day30 receipt because it contains administrative held-out aggregate audits.
Day33 remains the single frozen held-out final evaluation.

Scoring prompt SHA256: `f02290eb9fd0ad3de92363352fc921d13e9ce318de4d6db969149ec37fcd2cf7`.

Frozen outputs:
- `data/eval/day32_development_scoring_predictions.jsonl`
- `data/eval/day32_calibration_search.json`
- `data/eval/day32_development_calibrated_predictions.jsonl`
- `data/eval/day32_development_calibrated_metrics.json`
- `data/protocol/day32_frozen_diagnostic_config.json`
- `data/protocol/day32_calibration_freeze_receipt.json`
