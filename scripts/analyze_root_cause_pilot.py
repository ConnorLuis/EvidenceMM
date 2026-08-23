import json
from pathlib import Path
data=json.loads(Path('data/protocol/day23_pilot_summary.json').read_text())
for k,v in data['groups'].items():
    print(k, len(v))
