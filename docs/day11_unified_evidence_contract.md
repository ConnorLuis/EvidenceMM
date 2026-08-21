# EvidenceMM Day 11 - Unified Document + Robot Evidence Contract

## Goal

Day 11 starts the integration phase after the Day 7-10 temporal micro-baseline
series.

The goal is deliberately narrow:

> Represent real document evidence and real robot-operation evidence inside one
> traceable contract, then validate one cross-domain evidence packet without
> adding failure diagnosis, Agent orchestration, or new model inference.

This is the first structural bridge between the existing document grounded-QA
branch and the robot temporal-evidence branch.

## Existing gap

The generic `EvidenceRef` schema already supports PDF and `robot_sequence`
locators. However, the frozen Day 6 generation baseline still uses
PDF-specific `EvidencePage` and `CitationRef` objects.

Day 11 does not rewrite or invalidate the Day 6 baseline. It introduces a new
cross-domain contract alongside the frozen baseline so later integration can
migrate deliberately.

## Unified evidence bundle v1

The bundle schema version is:

```text
evidencemm_unified_evidence_bundle_v1
```

A bundle contains:

```text
question
+
one or more UnifiedEvidenceItem
```

Each item contains:

```text
evidence_id
kind
refs[]
provenance
payload
```

`refs[]` uses the existing generic `EvidenceRef` locator contract.

### Document page item

A document page item carries:

```text
PDF source_id
PDF source SHA256
source manifest path
1-based page number
normalized page-text SHA256
character count
text excerpt
optional rendered-page image path
```

Its citation is a PDF `EvidenceRef` with the matching page number.

### Robot sample item

A robot sample item is one canonical synchronized sample and carries:

```text
episode_id
episode SHA256
episode manifest path
samples.csv SHA256
metadata.json SHA256

frame_index
canonical elapsed timestamp

front original JPEG path + SHA256
wrist original JPEG path + SHA256

observation 6D
action 6D
tracking_error 6D
```

The two visual citations are `robot_sequence` EvidenceRefs for the same
frame/timestamp, one for `front` and one for `wrist`.

The state/action snapshot is attached to that same canonical sample identity;
Day 11 does not invent a second independent robot-state source.

## Provenance rules

Every item binds:

- source identity;
- source type;
- canonical SHA256;
- manifest path;
- optional supporting SHA256 values.

A citation ref must agree with the item's provenance source ID and source type.

Robot camera refs must agree with the robot payload frame and timestamp.

## Unified citation policy

Day 11 introduces a generic grounded-answer contract whose citations are
`EvidenceRef` rather than the Day 6 PDF-only citation shape.

The policy rejects:

- citations outside the supplied evidence bundle;
- duplicate citations;
- citations on abstention;
- an answerable response with zero citations.

This is a contract layer only. Day 11 Gate A does not yet replace the Day 6
Qwen3-VL prompt or generation pipeline.

## Minimal real cross-domain smoke

The smoke case binds two already-existing real sources:

```text
document
sts3215_datasheet
-> one real PDF page

robot
20260815_110415
-> one real synchronized frame
-> front + wrist original JPEGs
-> observation/action/tracking_error snapshot
```

Default smoke locators are:

```text
PDF page 1
robot frame 15
```

These defaults are chosen only as deterministic non-gold contract fixtures.
The smoke does not claim that the manual page explains the robot frame.

Before building the packet, the smoke verifies:

- PDF bytes against the source manifest SHA256;
- robot metadata and samples CSV against the episode manifest;
- original front/wrist frame bytes against FrameRecord SHA256;
- robot source semantics for observation/action;
- frame-record and state/action timestamp identity.

It then validates one citation set containing:

```text
PDF page citation
robot front frame citation
robot wrist frame citation
```

## Explicit non-claims

Day 11 Gate A does **not** claim:

- unified retrieval is implemented;
- unified reranking is implemented;
- document evidence explains the robot behavior;
- failure diagnosis is implemented;
- Qwen3-VL has consumed the cross-domain packet;
- Agent or MCP orchestration is involved;
- MP4 upload is implemented.

The only claim is that real document and robot evidence can now share one
validated, provenance-bound, citation-capable contract.

## Next gate

After the schema and real cross-domain smoke are frozen, the next gate can
connect this bundle to grounded generation while preserving citation
validation and abstention behavior.
