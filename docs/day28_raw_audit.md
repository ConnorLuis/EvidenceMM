# Day28 — Raw audit, exclusions, and source binding

## Scope

Day28 closes the acquisition-to-evidence provenance layer for Root-Cause Benchmark v2.

It does **not** assign Ground Truth, prejudge evidence answerability, perform Day29 human causal review, or materialize the Day30 pair-group split.

## Frozen inputs

Day28 consumes the already frozen Day24–Day27 record files. Historical records are not rewritten.

Frozen Day22 collection-plan SHA256:

`93345b1fd8330fa9e6076b95de018e423750788d91ccb26ac15258e92916e76d`

Day27 provenance commit:

`eaa29a3ebc9f41fa26ffa6de3291c6a28d93a4cd`

## Raw ledger

The final raw root contains 93 physical episode directories:

- 92 are registered in Day24–Day27 records;
- 90 registered attempts are selected canonical benchmark episodes;
- 2 registered attempts are retained noncanonical experimental exclusions;
- 1 additional directory is a technical exclusion.

The technical exclusion is audited without persisting its episode identity or raw relative path. Day28 binds it only by technical properties and immutable tree SHA256.

## Fresh technical re-audit

All 92 registered attempts are re-read using the existing `audit_episode()` contract. The registered record's technical fields must exactly match the fresh audit.

## Label-safe source manifest

`day28_registered_source_manifest.csv` contains source identity and technical provenance only. It deliberately excludes pair/plan identifiers, cause/intervention metadata, canonical-selection status, task outcome, experimental-validity status, Ground Truth fields, and review notes.

## Fingerprints

Each episode tree SHA256 binds relative file names, file sizes, and file bytes.

Registered 92-source aggregate:

`5ec4c38b8c4653781b77b9237951fbfb330541cbd7d607159fcf52e90c621a81`

Anonymous technical exclusion tree:

`ecec47d923656166e26fbd5e46c0c3b54c91702cdc55142e917137fac4175191`

Day28 also computes a 93-tree content-set SHA256 without encoding the anonymous directory name.

## Retention certification

`raw_attempt_retention_completeness_asserted=true` is allowed only when all 92 registered sources remain present and byte-consistent, exactly one anonymous technical exclusion matches the frozen fingerprint, and all 93 physical directories are accounted for.

## Storage migration

The compatibility path remains `/mnt/c/Users/Administrator/projects/embodied-agent-arm/outputs/episodes_root_cause_v2_final`. It may be backed by an NTFS Junction. Physical disk location is not benchmark identity; content fingerprints are.
