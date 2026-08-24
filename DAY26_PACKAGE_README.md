# EvidenceMM Day26 precollection package

Day26 scope: final `trajectory_execution_deviation` collection.

Frozen contract:

- source protocol: Day22 frozen collection plan (`93345b1f...e76d`)
- parameter source: Day23 frozen intervention parameters
- previous frozen commit: Day25 `2eb16ae1fb9418af0a7c712dc321b69fd3f0ed42`
- new trajectory episodes: 20
- primary slots: all 15 `s04`
- repeat slots: `G03/G06/G09/G12/G15` `s05`
- Day24 clean anchors: reuse 15/15, no new clean collection
- frozen deviation: Follower forward, 40–60 mm
- object pose: nominal
- gripper timing: nominal
- future split: must remain unmaterialized until Day30

After applying this package from the EvidenceMM repository root, run:

```bash
pytest -q tests/test_day26_trajectory_collection.py
pytest -q
python scripts/analyze_day26_trajectory_collection.py
python scripts/validate_day26_trajectory_collection.py
python scripts/generate_day26_recorder_commands.py
```

Expected precollection state:

- `trajectory_canonical=0/20`
- `trajectory_failure=0/20`
- `clean_anchors=15/15`
- `complete_groups=0/15`
- `new_clean_collection_required=False`
- analyzer status = `in_progress`
- validator = `NOT READY` only because 20 new trajectory episodes have not yet been collected

Do not start a new pilot, do not tune 40–60 mm, and do not recollect clean controls.
