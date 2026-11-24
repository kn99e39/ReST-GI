"""WL112 freeze manifest: SHA-256 of every WL112 source, the protocol, the imported unchanged ECRO / WL100 sources and inputs."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C
EXP = HERE.parent
files = sorted(p for p in HERE.glob('*.py')) + [HERE / 'chain112.sh', HERE / 'protocol' / 'sss112.json']
deps = ['ecro86/pt86.py', 'ecro86/scene86.py', 'ecro86/model86.py', 'ecro86/protocol/ecro86.json', 'ecro86/data/mats86.npz',
        'ecro_material100/pt100.py', 'ecro_material100/common100.py', 'ecro_material100/protocol.json',
        'fnto_compiled83/closure83_dr.py', 'fnto_compiled81/layered81.py']
inputs = ['results/ecro_material100/sss/model.pt', 'results/ecro_material100/configs.json',
          'results/ecro_material100/sss/renders/REF_0_0.npz', 'results/ecro_material100/sss/renders/REF_0_1.npz']
m = {'batch': 'WL112', 'base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
     'wl112_sources': {p.name: C.sha(p) for p in files}, 'unchanged_dependencies': {d: C.sha(EXP / d) for d in deps},
     'inputs': {i: C.sha(C.REPO / i) for i in inputs}}
C.jd(m, HERE / 'protocol' / 'freeze_manifest.json')
print(json.dumps(m['inputs'], indent=1))
