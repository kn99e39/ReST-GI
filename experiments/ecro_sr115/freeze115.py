"""WL115 freeze manifest: SHA-256 of WL115 sources, protocol, imported WL112-WL114 / ECRO sources and authenticated inputs.
python freeze115.py LABEL  -> protocol/freeze_<LABEL>.json"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C
EXP = HERE.parent
label = sys.argv[1] if len(sys.argv) > 1 else 'stageA'
files = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh')) + [HERE / 'protocol' / 'sr115.json']
deps = ['ecro_fd114/tracer114.py', 'ecro_fd114/common114.py', 'ecro_fd114/eval114.py', 'ecro_fd114/analyze114.py', 'ecro_fd114/train114.py',
        'ecro_fd114/data114.py', 'ecro_sss113/tracer113.py', 'ecro_sss113/common113.py', 'ecro_sss113/metrics113.py', 'ecro_sss113/eval113.py',
        'ecro_sss113/train113.py', 'ecro_sss113/model113.py', 'ecro_sss112/tracer112.py', 'ecro_sss112/common112.py', 'ecro_sss112/metrics112.py',
        'ecro86/pt86.py', 'ecro86/scene86.py', 'ecro86/model86.py', 'ecro_material100/pt100.py', 'ecro_material100/common100.py',
        'fnto_compiled83/closure83_dr.py']
inputs = ['results/ecro_material100/configs.json', 'results/ecro_fd114/primary/stageC/B1/model.pt', 'results/ecro_fd114/primary/stageC/B2/model.pt',
          'results/ecro_fd114/primary/stageC/A1r/model.pt', 'results/ecro_fd114/primary/stageC/A2/model.pt', 'results/ecro_fd114/primary/stageC/OB1/model.pt',
          'results/ecro_fd114/primary/analysis/stageA_qualification.json']
m = {'batch': 'WL115', 'label': label, 'base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
     'wl115_sources': {p.name: C.sha(p) for p in files if p.exists()}, 'unchanged_dependencies': {d: C.sha(EXP / d) for d in deps},
     'inputs': {i: C.sha(C.REPO / i) for i in inputs}}
C.jd(m, HERE / 'protocol' / f'freeze_{label}.json'); print(json.dumps(m['inputs'], indent=1))
