# Day30 Groupwise Development / Held-out Split Freeze

## Purpose

Day30 materializes the future split that was frozen prospectively in the Day22 protocol.

The split rule is not selected or tuned on Day30. Day30 only evaluates the already-frozen rule:

- unit: `pair_group_id`
- seed: `evidencemm-root-cause-v2-split-v3`
- ranking: `sha256(seed|pair_group_id)` ascending
- first 10 ranked groups: `development`
- final 5 ranked groups: `held_out`
- 60 development episodes / 30 held-out episodes
- no pair group may cross the split

Day29 Ground Truth must already be frozen before this phase begins.

## Deterministic ranking

| Rank | Pair group | SHA-256 | Split |
| ---: | --- | --- | --- |
| 1 | `rcv2_g07` | `02d7261b78b30ff384de33423002fe767e1b10abaaa7b78eba529c994b02020c` | `development` |
| 2 | `rcv2_g11` | `2884ac16a06fe4ef29e8743eceaaa0e0fd4c820e5e8cf18f5cad05d836ab2a55` | `development` |
| 3 | `rcv2_g15` | `31002567c45aa5f43f674bc32f815a9ec6e58e76484638ebe453f54ea2be9e12` | `development` |
| 4 | `rcv2_g05` | `33fcedf2369e9c35fdc06ab06704bc82c32ae3630c92fc31a7d6a28f8c04d5bd` | `development` |
| 5 | `rcv2_g01` | `486f844e1fc1e4486ee6228b72293a2c40078e73e6b818ee339855b98c00bc90` | `development` |
| 6 | `rcv2_g10` | `4d7b09675d6be144a2d2e41a5d57f0a6de340fb57be7644cfc98d227397142df` | `development` |
| 7 | `rcv2_g14` | `5032e9c452bb582571fe9fd02b5e179db19975527abfbe86109c24e7c87f551e` | `development` |
| 8 | `rcv2_g13` | `50d4059dc42c451f7875a09bce9162a118abc6c729d5c704dbd91d6ba38e9406` | `development` |
| 9 | `rcv2_g06` | `52c7de2d870f81cec826d5a1b673c6b9fa95fd1cbe758e07412ffb0ac32cc793` | `development` |
| 10 | `rcv2_g04` | `6de06c8a010ca604ee6812982d129593a31d2000db11f3dc96864f1e2b6ce305` | `development` |
| 11 | `rcv2_g08` | `72ff3a0251e1f3f3a8089b26676449b5fc2d9105a89370e4acd01390a4261c1e` | `held_out` |
| 12 | `rcv2_g09` | `75fd43af61986e03e4766c5ffd667b527e03913c9b2dbb85357053c8914eccc9` | `held_out` |
| 13 | `rcv2_g12` | `9ac3fc1d481de1b47e6e6d7619d4028897a0749f6f8f1c01f7a4f5344c38c015` | `held_out` |
| 14 | `rcv2_g02` | `b1187bb1f9cbf1f0784546a918e31dcdfa845bd72b29086968462b3149cde395` | `held_out` |
| 15 | `rcv2_g03` | `f33196350e1287b56beda3953cbc75f7c03eb9a86a09ff1d7191624ba8375ec3` | `held_out` |

## Expected membership

Development:

`rcv2_g07, rcv2_g11, rcv2_g15, rcv2_g05, rcv2_g01, rcv2_g10, rcv2_g14, rcv2_g13, rcv2_g06, rcv2_g04`

Held-out:

`rcv2_g08, rcv2_g09, rcv2_g12, rcv2_g02, rcv2_g03`

This membership is a cryptographic consequence of the Day22 seed and group IDs. It is not
changed to balance labels.

## Tool commands

```bash
python scripts/day30_split.py preflight
python scripts/day30_split.py materialize
python scripts/day30_split.py validate
python scripts/day30_split.py freeze
python scripts/day30_split.py audit
```

## Outputs

`data/splits/day30_pair_group_split.json`

- ranked pair groups
- deterministic hash
- split assignment
- no physical-cause or diagnostic labels

`data/splits/day30_episode_split.jsonl`

- one row per canonical episode
- `episode_id`
- `pair_group_id`
- pair-group rank/hash
- development / held-out membership
- no Ground Truth labels

`data/protocol/day30_split_freeze_receipt.json`

- source hashes
- split artifact hashes
- counts and group isolation checks
- post-assignment label distribution for audit only
- confirms the ranking did not use labels

## Held-out policy

Day31 and Day32:

- held-out model selection: forbidden
- held-out prompt tuning: forbidden
- held-out retrieval tuning: forbidden
- held-out calibration: forbidden
- held-out labels must not participate in development decisions

Day33:

- one frozen held-out final evaluation

Day30 itself performs no model training, calibration, prompt tuning, retrieval tuning, or held-out evaluation.

## Git closure

Tooling commit:

```bash
git add   scripts/day30_split.py   data/protocol/day30_split_operational_contract.json   docs/day30_split_operator_guide.md   tests/test_day30_split.py

git commit -m "feat(day30): freeze groupwise split tooling"
git push origin master
```

After materialize + validate + freeze + audit:

```bash
git add   data/splits/day30_pair_group_split.json   data/splits/day30_episode_split.jsonl   data/protocol/day30_split_freeze_receipt.json

git commit -m "feat(day30): freeze pair group development heldout split"
git push origin master
```

A successful post-commit `python scripts/day30_split.py audit` and clean worktree close Day30.
