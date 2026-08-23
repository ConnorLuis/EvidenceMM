import json
from pathlib import Path
d=json.loads(Path('data/protocol/day23_pilot_summary.json').read_text())
assert len(d['groups']['clean'])==3
assert len(d['groups']['target_offset_or_perception'])==3
assert len(d['groups']['gripper_close_timing'])==3
assert len(d['groups']['trajectory_execution_deviation'])==3
print({'valid': True, 'status':'DAY23 ROOT CAUSE BENCHMARK PASS'})
