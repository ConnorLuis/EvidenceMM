# EvidenceMM Day25 complete package

Apply from the EvidenceMM repository root:

```bash
python /path/to/apply_day25_package.py --repo .
```

Then run:

```bash
pytest -q
python scripts/generate_day25_recorder_commands.py
python scripts/analyze_day25_gripper_collection.py
python scripts/validate_day25_gripper_collection.py
```

Precollection expected state:

- `gripper_canonical=0/20`
- `clean_anchors=15/15`
- `complete_groups=0/15`
- validator = `NOT READY` only because the 20 new gripper episodes are not collected yet.

Day25 deliberately reuses Day24 clean anchors and does not recollect 15 clean episodes.
