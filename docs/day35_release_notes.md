# Day35 Release Notes

## Release character

This is a staged research-project closure, not a production release.

## Major additions in the completed project

- traceable document and robot multimodal evidence contracts;
- BM25, BGE-M3, reranking and visual-retrieval components;
- Qwen3-VL grounded generation;
- synchronized SO-ARM101 front/wrist + state/action evidence;
- frozen root-cause benchmark protocol;
- 90 canonical final episodes / 15 pair groups;
- frozen 60-development / 30-held-out groupwise split;
- zero-shot diagnostic baseline;
- development-only calibration;
- exactly one frozen held-out final evaluation;
- post-hoc confusion/error/generalization analysis;
- latency, GPU and development-only E2E profiling;
- final benchmark/reproducibility/limitations documentation.

## Final benchmark headline

The single frozen held-out result is intentionally not rewritten:

- Answerable 3-class Macro-F1: 0.1672
- Substantive 4-class Macro-F1: 0.1157
- Overall accuracy: 0.1667
- Parse rate: 0.8333

The most important diagnosed weaknesses are calibration-induced gripper-class
collapse, clean-control over-diagnosis and JSON structured-output failures.

## Release policy

Day22-Day35 artifacts are frozen. New accuracy-improving work must start in a
new phase with a new protocol rather than modifying this release history.
