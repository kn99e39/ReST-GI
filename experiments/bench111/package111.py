"""WL111 reviewer package: copy figures and summary into evaluation/111/ with a manifest (sha256 + source).
python package111.py FIG_DIR SUMMARY_JSON EVAL_DIR"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

FIG, SUMF, EVAL = (Path(a) for a in sys.argv[1:4])
EVAL.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


items = []
for p in sorted(FIG.glob('*.png')) + [SUMF]:
    dst = EVAL / p.name
    shutil.copy2(p, dst)
    items.append({'file': p.name, 'sha256': sha(dst), 'source': str(p), 'generator': 'experiments/bench111/figs111.py' if p.suffix == '.png'
                  else 'experiments/bench111/analyze111.py'})
man = {'worklog': 'docs/worklogs/111_WORKLOG_EXTERNAL_BASELINE_BENCHMARK_AND_APPROVAL_GATE.md', 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
       'display_transform': 'LDR = clip(x, 0, 1)^(1/2.2), exposure 1 (all image panels); error maps signed relative luminance, shared scale per figure',
       'items': items}
(EVAL / 'manifest111.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
print('PACKAGED', len(items))
