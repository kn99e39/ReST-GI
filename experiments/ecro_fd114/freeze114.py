"""WL114 freeze manifest: SHA-256 of WL114 sources, protocol, imported WL112 / WL113 / ECRO sources, authenticated inputs."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common114 as C
EXP = HERE.parent
files = sorted(HERE.glob('*.py')) + [HERE / 'chain114.sh', HERE / 'protocol' / 'fd114.json']
deps = ['ecro_sss113/tracer113.py', 'ecro_sss113/common113.py', 'ecro_sss113/metrics113.py', 'ecro_sss113/eval113.py', 'ecro_sss113/train113.py',
        'ecro_sss113/model113.py', 'ecro_sss112/tracer112.py', 'ecro_sss112/common112.py', 'ecro_sss112/metrics112.py', 'ecro_sss112/stageA112.py',
        'ecro86/pt86.py', 'ecro86/scene86.py', 'ecro86/model86.py', 'ecro_material100/pt100.py', 'ecro_material100/common100.py',
        'fnto_compiled83/closure83_dr.py']
inputs = ['results/ecro_material100/configs.json', 'results/ecro_sss113/primary/stageC/CAND/model.pt', 'results/ecro_sss113/primary/stageC/O_U/model.pt',
          'results/ecro_fd114/design/design_check.json']
ext = {'Ref-65 static_V0.npz': r'C:\Users\dna10\TransGI-Reproduction-data\runs\living_room_3\reference\static_V0.npz',
       'Ref-7 static_V0_d7.npz': r'C:\Users\dna10\TransGI-Reproduction-data\runs\living_room_3\reference\static_V0_d7.npz',
       'TransGI reference.py': r'C:\Projects\TransGI_Reproduction\transgi\reference.py'}
m = {'batch': 'WL114', 'base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
     'wl114_sources': {p.name: C.sha(p) for p in files}, 'unchanged_dependencies': {d: C.sha(EXP / d) for d in deps},
     'inputs': {i: C.sha(C.REPO / i) for i in inputs}, 'external_inputs': {k: C.sha(Path(v)) for k, v in ext.items()}}
C.jd(m, HERE / 'protocol' / 'freeze_manifest.json'); print(json.dumps(m['inputs'], indent=1)); print(json.dumps(m['external_inputs'], indent=1))
