# EvidenceMM

EvidenceMM is a traceable multimodal evidence system for technical documents and
real robot-operation sequences. The repository combines provenance-preserving
retrieval/grounding components with a frozen SO-ARM101 robot-failure benchmark
and a Qwen3-VL diagnostic baseline.

> **Project status:** staged research prototype complete and frozen through Day35.
> The repository is interview-ready and reproducible from the committed benchmark
> manifests and audit scripts. It is not claimed as a production system or a
> high-accuracy root-cause classifier.

## What is implemented

### Multimodal evidence system

- PDF/image source binding with SHA256 provenance;
- BM25 sparse retrieval;
- BGE-M3 dense retrieval;
- sparse+dense candidate union;
- BGE cross-encoder reranking with ranking traces;
- ColQwen2.5 visual page retrieval as a validated separate component;
- Qwen3-VL grounded generation;
- deterministic compact-citation resolution;
- citation/required-fact validation and fail-closed abstention;
- `UnifiedEvidenceBundle` for document and robot evidence;
- pipeline-contract failure diagnosis.

### Robot-operation evidence

The canonical robot source binds:

```text
metadata.json
samples.csv
front/*.jpg
wrist/*.jpg
```

Each robot evidence unit is traceable to episode, frame, timestamp, synchronized
front/wrist images, observation, action and tracking error.

### Root-cause benchmark v2

The final benchmark contains **90 canonical episodes in 15 pair groups**. The
frozen Day30 split is groupwise:

- development: **60 episodes / 10 pair groups**;
- held-out: **30 episodes / 5 pair groups**;
- pair-group crossing: **0**.

The diagnostic taxonomy is:

- `target_offset_or_perception`;
- `gripper_close_timing`;
- `trajectory_execution_deviation`;
- `insufficient_evidence`;
- `clean_success`.

The benchmark was protocol-frozen before final collection/evaluation. Human
causal ground truth was frozen before the development/held-out split, Day32
calibration used development data only, and Day33 consumed the held-out final
evaluation exactly once.

## Final benchmark result

Day31 established the untuned baseline: all 60 development episodes parsed, but
all 60 became `insufficient_evidence`.

Day32 separated substantive scoring from abstention and calibrated only on
development pair groups.

Day33 then ran the single frozen held-out evaluation. Day34 independently
recomputed and audited the result.

| Metric | Frozen held-out |
| --- | ---: |
| Answerable 3-class Macro-F1 | **0.1672** |
| Failed-case 4-way Macro-F1 | **0.1254** |
| Substantive 4-class Macro-F1 | **0.1157** |
| Abstention accuracy | **0.8800** |
| False-abstention rate | **0.1200** |
| Clean-control accuracy | **0.0000** |
| Overall decision accuracy | **0.1667** |
| Structured-output parse rate | **0.8333** |

These are intentionally reported as the frozen result; no post-held-out tuning
or regeneration was performed.

## What the final result exposed

The project preserves negative results rather than hiding them.

### 1. Development-to-held-out generalization gap

Answerable 3-class Macro-F1 fell from **0.3185** on the full development split
to **0.1672** on the single held-out evaluation.

### 2. Calibration suppressed the gripper class

Among 25 parse-success held-out cases, raw Qwen3-VL scoring ranked
`gripper_close_timing` first **14** times. After the frozen Day32 calibration,
the final classifier predicted `gripper_close_timing` **0** times. The held-out
ground truth contains 9 gripper cases, so final gripper recall is **0.0**.

### 3. Structured-output reliability is a system bottleneck

Five of 30 held-out responses failed JSON parsing. All five were fail-closed to
`insufficient_evidence`; no result-conditioned retry was allowed.

### 4. Clean controls are not reliably separated from failures

There are 5 held-out `clean_success` controls and the frozen model identifies
none correctly. Three are over-diagnosed as physical causes and two fail closed.

## Efficiency

Frozen Day33 held-out generation-only latency:

- mean: **10.040 s**;
- P50: **9.738 s**;
- P95: **13.366 s**;
- mean peak allocated GPU memory: **9427.9 MB**.

A separate Day34 **development-only**, GT-blind E2E profile over five
deterministically selected episodes measured:

- warm E2E mean: **9.194 s**;
- warm E2E P95: **10.122 s**;
- model load: **7.893 s**;
- processor load: **0.211 s**;
- throughput: **6.53 episodes/minute**.

The root-cause input reduces a 900-frame episode to 12 evidence frames:

- evidence density: **1.333%**;
- frame reduction: **98.667%**;
- frame-count compression: **75x**.

Latency is hardware-specific; the frozen Day34 profile was run on an NVIDIA
GeForce RTX 4070 SUPER.

## Experimental integrity

The final benchmark chain is intentionally versioned:

```text
Day22  protocol freeze
Day23  excluded pilot + intervention freeze
Day24  target-offset collection
Day25  gripper-timing collection
Day26  trajectory-deviation collection
Day27  insufficient-evidence challenge
Day28  raw audit + source provenance
Day29  human causal ground truth
Day30  groupwise development/held-out split
Day31  zero-shot diagnostic baseline
Day32  development-only calibration
Day33  single frozen held-out final evaluation
Day34  metrics/error/latency/GPU/E2E analysis
Day35  project/release/documentation closure
```

Key integrity properties:

- the Day30 split is groupwise and label-blind at materialization;
- Day31 uses development only;
- Day32 uses development labels only;
- Day33 authorization is committed and pushed before first held-out inference;
- held-out final-evaluation count consumed: **1**;
- Day34 held-out inference/regeneration count: **0**;
- no prompt, calibration, retrieval or model selection after Day33;
- frozen predictions were not rewritten after results were known.

See [`docs/day35_benchmark_card.md`](docs/day35_benchmark_card.md) for the
benchmark card and [`docs/day35_reproducibility.md`](docs/day35_reproducibility.md)
for audit/reproduction instructions.

## Architecture

```text
                         EvidenceMM
                             |
             +---------------+---------------+
             |                               |
       Document evidence                Robot evidence
             |                               |
       BM25 / BGE-M3                 front + wrist images
       / reranking                   state / action / error
             |                               |
             +---------------+---------------+
                             |
                   traceable evidence bundle
                             |
                          Qwen3-VL
                             |
               structured evidence scoring
                             |
                 frozen calibration / rules
                             |
                diagnosis or fail-closed
                             |
          metrics + per-case trace + audit receipt
```

The document/RAG branch and the root-cause-v2 benchmark are related evidence
system components, but the Day31-Day33 root-cause baseline itself uses no manual
retrieval corpus.

## Reproducibility

Install:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Core frozen audits:

```bash
python scripts/day31_root_cause_baseline.py audit
python scripts/day32_development_calibration.py audit
python scripts/day33_single_heldout_eval.py audit
python scripts/day34_metrics_error_efficiency.py audit
python scripts/day35_release_audit.py audit
```

Large raw robot media and local model caches are intentionally not tracked by
Git. The repository keeps source manifests, hashes, frozen annotations,
split/evaluation artifacts and protocol receipts; local raw-data bindings are
validated by the audit pipeline.

## Important limitations / non-claims

EvidenceMM is **not** claimed to be:

- a production robot monitoring service;
- a high-accuracy physical root-cause classifier;
- an Agent/MCP orchestration system;
- a benchmark where temporal localization and grounding metrics should be
  silently merged with the final root-cause result.

The Day20 temporal interval experiment remains a separate historical benchmark.
The Day31-Day33 root-cause predictor does not emit a failure interval and uses
no retrieval/manual corpus, so Day34 correctly reports temporal localization as
not emitted and grounding as not applicable for this final baseline.

See [`docs/day35_limitations.md`](docs/day35_limitations.md).

## Project boundary

```text
chat-api   = LLM / RAG gateway
agent-api  = Agent reasoning / orchestration
EvidenceMM = multimodal evidence retrieval / grounding / diagnosis
SO-ARM101  = robot data / imitation-learning validation
```

EvidenceMM deliberately does not duplicate LangGraph, MCP, memory or planning
from the Agent project.

## Release state

Day35 freezes this repository as an **interview-ready research prototype**.
Future work should start from a new experimental phase rather than rewriting
the Day22-Day35 benchmark history.
