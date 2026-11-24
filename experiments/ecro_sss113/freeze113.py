"""WL113 freeze manifest: SHA-256 of WL113 sources, protocol, imported WL112 / ECRO sources and inputs."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C
EXP = HERE.parent
files = sorted(HERE.glob('*.py')) + [HERE / 'chain113.sh', HERE / 'protocol' / 'sss113.json']
deps = ['ecro_sss112/tracer112.py', 'ecro_sss112/common112.py', 'ecro_sss112/metrics112.py', 'ecro_sss112/analyzeAB112.py', 'ecro_sss112/stageA112.py',
        'ecro86/pt86.py', 'ecro86/scene86.py', 'ecro86/model86.py', 'ecro_material100/pt100.py', 'ecro_material100/common100.py', 'fnto_compiled83/closure83_dr.py']
inputs = ['results/ecro_material100/sss/model.pt', 'results/ecro_material100/configs.json']
m = {'batch': 'WL113', 'base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
     'wl113_sources': {p.name: C.sha(p) for p in files}, 'unchanged_dependencies': {d: C.sha(EXP / d) for d in deps},
     'inputs': {i: C.sha(C.REPO / i) for i in inputs}}
C.jd(m, HERE / 'protocol' / 'freeze_manifest.json'); print(m['inputs'])
