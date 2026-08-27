# Day33 Single Frozen Held-Out Final Evaluation

Day33 is the single final held-out diagnostic evaluation required by the Day22 roadmap.

The Day32 configuration is immutable: same Qwen3-VL model, exact Day32 scoring prompt,
exact frozen class log-biases, and exact thresholds. No prompt, calibration, retrieval,
evidence-selection, or model-selection changes are allowed after authorization.

Day31 never produced held-out predictions, so the phrase
`day31_selected_frame_indices_reused` is operationalized before any held-out inference as
re-applying the exact frozen Day31 deterministic selection rule to held-out episodes:
7 uniform anchors + 5 dynamic frames, 20-frame minimum separation, 12 total frames,
same front/wrist contact-sheet convention, and same state/action text convention.

Before the first held-out inference, an authorization receipt is committed and must be
pushed to remote master. That consumes the single held-out evaluation count. Operational
interruptions may resume the same deterministic run; result-conditioned regeneration,
parser-failure retry, configuration edits, or a second fresh held-out run are forbidden.

The scoring stage does not parse GT. Only after all 30 predictions are complete and
validated does evaluation parse held-out GT rows. Development rows are skipped before
JSON parsing.

Final outputs:
- data/protocol/day33_heldout_eval_start_receipt.json
- data/eval/day33_heldout_scoring_predictions.jsonl
- data/eval/day33_heldout_final_predictions.jsonl
- data/eval/day33_heldout_final_metrics.json
- data/protocol/day33_heldout_eval_freeze_receipt.json
