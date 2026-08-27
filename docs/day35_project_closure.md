# Day35 Project Closure

## Closure decision

EvidenceMM is **stage-complete and interview-ready** as a research engineering
project.

The correct project statement is:

> EvidenceMM is a traceable multimodal evidence system that integrates document
> retrieval/grounding and synchronized robot-operation evidence, then builds a
> protocol-frozen real-robot root-cause benchmark with development-only
> calibration, a single held-out evaluation, and post-hoc failure/efficiency
> analysis.

It should **not** be described as a high-accuracy robot root-cause classifier.

## What was completed

### Evidence infrastructure

The repository contains provenance-preserving document and robot evidence,
sparse/dense/visual retrieval components, grounded generation contracts,
citation validation, unified evidence bundles and pipeline-failure diagnosis.

### Real robot benchmark

The project collected and audited final real SO-ARM101 operation sequences for
target-offset, gripper-timing, trajectory-deviation and clean/answerability
controls.

Ground truth, groupwise split, baseline, calibration and held-out evaluation
were versioned as separate frozen stages.

### Final evaluation

The final held-out result was accepted on the first authorized run rather than
tuned after observation. Day34 then quantified the main failure modes:

- cause confusion: 15/30;
- parse-failure abstention: 5/30;
- clean false-positive cause: 3/30;
- failure predicted clean: 2/30;
- correct: 5/30.

### Efficiency

The system uses 12 selected evidence frames per 900-frame episode and records
generation latency, GPU peak memory and a separate development-only E2E profile.

## Why this is interview-worthy despite low accuracy

The project demonstrates engineering and experimental skills that a high
development-set score alone does not:

1. source-byte provenance and traceability;
2. multimodal evidence construction;
3. real robot data handling;
4. protocol precommit and groupwise splitting;
5. development/held-out separation;
6. fail-closed structured-output handling;
7. post-hoc error analysis without contaminating held-out;
8. latency/GPU/E2E profiling;
9. preserving negative results and identifying calibration shift.

## Future work boundary

The frozen benchmark suggests several next hypotheses:

- constrained/schema-native structured decoding;
- calibration methods robust to class-distribution shift;
- explicit clean/negative evidence modeling;
- larger groupwise benchmark;
- retrieval/manual evidence integration for physical diagnosis;
- temporal interval + causal diagnosis in one precommitted benchmark.

These are **future experiments**, not fixes to be applied retroactively to the
Day33 held-out result.

## Final phase state

Day22-Day35 should now be treated as historical frozen project state.
Subsequent work should branch conceptually into a new benchmark version or new
project phase.
