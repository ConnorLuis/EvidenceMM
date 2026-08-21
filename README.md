# EvidenceMM

EvidenceMM is a traceable multimodal RAG system for complex documents and
robot-operation sequences.

The project is developed as a failure-driven series of reproducible baselines.
By the end of Day 7, the document branch has an end-to-end grounded-QA
baseline, while the robot branch has a separate temporal-evidence baseline.
Those two branches are **not yet connected into one cross-domain diagnostic
pipeline**.

## Current status

- latest completed feature commit: `04c7f1b`
  (`feat(temporal): add robot sequence evidence baseline`)
- Day 1-7 capability chain is complete
- Day 7 closure test suite: 37 tests passed
- Day 7.5 public-contract/schema calibration adds one schema-contract test
- canonical robot source: sample-synchronized front/wrist image sequence
- cross-document + robot evidence generation: **not yet integrated**
- failed-grasp diagnosis: **not yet implemented**
- current evaluation remains a smoke baseline, not a production benchmark

## Architecture

```text
                         EvidenceMM
                             |
              +--------------+--------------+
              |                             |
      document evidence                robot evidence
              |                             |
      PDF / page images          metadata.json + samples.csv
              |                    front/*.jpg + wrist/*.jpg
              |                             |
      SourceManifest / SHA256          EpisodeManifest
              |                        FrameRecord
              |                             |
       +------+-------+             TemporalSlice
       |              |             midpoint baseline
      BM25        ColQwen2.5               |
       |              |              temporal gold
       +------RRF-----+                    |
              |                       event coverage
      Top-k evidence pages
              |
    page image + extracted text
              |
       Qwen3-VL-4B-Instruct
              |
       structured grounded answer
              |
      citation validation + abstention
```

The document grounded-QA branch is connected end to end. The robot temporal
branch currently stops at evidence selection/evaluation. A later phase will
connect robot evidence with document evidence for diagnostic generation.

## Day 1-7 progression

| Day | Commit | Capability | Frozen observation |
| --- | --- | --- | --- |
| 1 | `650ce87` | Direct Qwen3-VL baseline + initial schemas/question bank | Direct static-image inference can speculate about temporal state |
| 2 | `f87a173` | Real PDF/image binding, source identity, SHA256, verified evidence | Evidence is explicitly traceable to bound source bytes |
| 3 | `725fd76` | Page-level BM25 text retrieval | On 2 verified PDF queries, one gold page is Rank 2 |
| 4 | `c055283` | ColQwen2.5 page-image multi-vector retrieval | Same gold page improves from BM25 Rank 2 to visual Rank 1 |
| 5 | `0402ba9` | BM25 + ColQwen2.5 Reciprocal Rank Fusion | RRF preserves both visual Top-1 hits without mixing incomparable raw scores |
| 6 | `49d2654` | Grounded generation, citation policy, structured output, abstention | 3/3 smoke cases pass the deterministic grounded-answer contract |
| 7 | `04c7f1b` | Synchronized robot-sequence temporal evidence baseline | 2 s uniform midpoint covers 2/3 verified events and misses a 0.268 s lift |

## Document branch

### Retrieval

Day 3 text-only BM25 over one bound 8-page STS3215 datasheet and two
human-verified PDF queries:

| Metric | BM25 |
| --- | ---: |
| Recall@1 | 0.5000 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| MRR@5 | 0.7500 |
| nDCG@5 | 0.8155 |

Day 4 ColQwen2.5 vision-only retrieval and Day 5 RRF hybrid retrieval both
reach perfect retrieval metrics on the same **two-query smoke set**. These
numbers validate the pipeline and the case-level failure mechanism; they are
not general retrieval-quality claims.

### Grounded generation

Day 6 connects hybrid retrieval to Qwen3-VL-4B-Instruct:

```text
question
   -> BM25 + ColQwen2.5
   -> RRF
   -> Top-2 evidence pages
   -> page image + extracted text
   -> Qwen3-VL
   -> strict structured answer
   -> citation validation
   -> answer or abstain
```

Observed deterministic metrics on **three smoke cases**:

| Metric | Value |
| --- | ---: |
| Structured output rate | 1.0000 |
| Answerability accuracy | 1.0000 |
| Citation-policy valid rate | 1.0000 |
| Citation gold-page hit rate | 1.0000 |
| Mean citation precision | 1.0000 |
| Mean required-fact coverage | 1.0000 |
| Abstention accuracy | 1.0000 |
| End-to-end pass rate | 1.0000 |

The perfect values above are explicitly limited to two answerable PDF cases
plus one controlled unsupported/abstention case.

## Robot temporal branch

Day 7 uses the original robot-operation sequence as canonical evidence:

```text
metadata.json
samples.csv
front/*.jpg
wrist/*.jpg
```

It does **not** manufacture an MP4 and then re-extract frames.

For episode `20260815_110415`:

- 900 sample-synchronized front/wrist pairs
- 1800 `FrameRecord` entries
- canonical timeline: `samples.csv:elapsed_ns`
- camera-specific source timestamp and source age retained
- original JPEG SHA256 retained
- deterministic aggregate episode SHA256
- joint state/action remain in the hashed source CSV but are intentionally
  excluded from the Day 7 visual baseline

The frozen temporal selector uses non-overlapping 2 s timestamp windows and
chooses the real sample nearest each window midpoint.

Observed selector diagnostics:

- 30 temporal slices
- mean midpoint error: 4.157 ms
- max midpoint error: 5.591 ms
- no image re-encoding
- original frame hashes reused

Human-verified visual temporal gold:

| Event | Inclusive frame interval | Duration | Uniform midpoint |
| --- | ---: | ---: | --- |
| `object_lift` | 408-412 | 0.268 s | MISS |
| `object_transport` | 413-530 | 7.802 s | HIT |
| `object_place` | 630-668 | 2.534 s | HIT |

Day 7 event coverage:

```text
2 / 3 = 0.6667
```

The short lift miss is deliberately preserved as a real baseline failure.

## Evidence schema

`src/evidencemm/schemas.py` defines the generic evidence locator contract for
single-file/document/image/video sources and robot-sequence references.

The canonical multi-file robot episode identity is defined separately by
`src/evidencemm/temporal_evidence.py::EpisodeManifest`, because a robot
sequence is composed of metadata, samples, and multiple camera frames rather
than one video path.

A robot-operation sequence may be represented by:

- sample-synchronized image sequences, as implemented in Day 7; or
- a native video source in a future adapter.

MP4 derived from the Day 7 JPEG sequence is display-only and is not canonical
evidence.

## Current evaluation scale

The repository is still in smoke-baseline mode:

- 1 bound 8-page PDF
- 2 human-verified PDF retrieval queries
- 3 Day 6 grounded-generation smoke cases
- 1 human-annotated robot episode
- 3 verified visual temporal events

Do not report the current perfect document metrics as general model quality.

## Explicit integration boundary

Already connected:

```text
document source
-> retrieval
-> hybrid evidence ranking
-> grounded generation
-> citation validation / abstention
```

Established but still separate:

```text
robot sequence
-> frame/evidence binding
-> temporal slicing
-> temporal event evaluation
```

Not yet connected:

```text
robot temporal evidence
+ manual/document evidence
-> cross-domain diagnostic answer
```

The flagship failed-grasp diagnosis remains a target, not a completed claim.

## Day 8 result

Day 8 compares a deterministic visual-motion-aware selector against the frozen
Day 7 uniform-midpoint baseline under the same episode, temporal windows,
verified gold, and 30-shared-sample / 60-image evidence budget.

The visual-motion rule is frozen as grayscale 160 x 120 adjacent-frame mean
absolute pixel difference, fused with `max(front, wrist)` and lower-frame-index
tie breaking.

Observed one-episode smoke result:

| Metric | Uniform midpoint | Visual motion |
| --- | ---: | ---: |
| Event coverage | 0.6667 | 0.6667 |
| Mean closest-evidence distance | 344.467 ms | 544.470 ms |

The motion selector does not improve event coverage and worsens mean temporal
proximity by 200.003 ms. It improves `object_transport` substantially but still
misses the 0.268 s `object_lift` event. This negative result is frozen without
post-hoc tuning.

The next independent comparison will introduce robot state/action evidence;
the Day 8 visual-motion parameters will remain unchanged.

## Day 9 result

Day 9 adds a robot-state/action-aware temporal selector using the canonical
`samples.csv` control signals while preserving the same episode, frozen
two-second windows, verified gold, and 30-shared-sample / 60-image evidence
budget.

The source metadata defines `observation_*` as the follower
`Present_Position` read before the current action write and `action_*` as the
final absolute `Goal_Position` actually sent after mapping, clamp and rate
limiting. The frozen selector uses equal-weight 6D adjacent RMS changes:

```text
state_change(t)  = RMS(q_t - q_(t-1))
action_change(t) = RMS(a_t - a_(t-1))
score(t)         = max(state_change(t), action_change(t))
```

Observed one-episode three-way smoke result:

| Metric | Midpoint | Visual motion | State/action |
| --- | ---: | ---: | ---: |
| Event coverage | 0.6667 | 0.6667 | 0.6667 |
| Mean closest-evidence distance | 344.467 ms | 544.470 ms | 699.200 ms |

The state/action selector still misses the 0.268 s `object_lift` event and is
worse than both earlier baselines on mean temporal proximity. Of its 30
selected windows, 27 are action-dominated and 3 are state-dominated.

This negative result is frozen without post-hoc joint weighting, gripper
boosting, normalization or threshold tuning. State/action remains valuable as
diagnostic evidence, but simple within-window change magnitude does not solve
the current short-event localization problem.

## Project boundaries

EvidenceMM is the multimodal perception/evidence layer.

- model-access concerns belong to `chat-api`
- reasoning/orchestration concerns belong to `agent-api`
- robot policy/control belongs to the SO-ARM101 / LeRobot project
- EvidenceMM focuses on evidence identity, retrieval, temporal localization,
  grounding, citation, abstention, and later multimodal diagnosis

See `docs/task_definition.md` and the Day-specific documents for the detailed
contracts and frozen smoke results.
