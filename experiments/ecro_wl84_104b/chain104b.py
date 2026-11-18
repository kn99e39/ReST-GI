"""WL104B chain: the three image workers in parallel (one process each, ~3 GB VRAM each), then the analysis.
  <venv-cu128 python> chain104b.py <raw_root> <reviewer_dir>
Success is judged by completion artefacts (DONE per image, index.html), not exit codes (WL103: os._exit aborts on this stack).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_wl84_104'))
import common104 as C  # noqa: E402

ROOT = Path(sys.argv[1]).resolve(); EVAL = Path(sys.argv[2]).resolve()
ROOT.mkdir(parents=True, exist_ok=True); (ROOT / 'logs').mkdir(exist_ok=True)
ST = ROOT / 'chain.json'
SMOKE = os.environ.get('WL104B_SMOKE') == '1'
dirty = subprocess.check_output(['git', '-C', str(C.REPO), 'status', '--porcelain', '--', 'experiments/ecro_wl84_104b'], text=True).strip() != ''
status = {'commit': subprocess.check_output(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], text=True).strip(), 'smoke': SMOKE,
          'experiment_dir_dirty': dirty, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
save = lambda: C.write_json(ST, status)
if dirty and not SMOKE:
    status['stopped'] = 'dirty_experiment_dir'; save(); sys.exit(2)
save()
procs = {}
for im in C.IMAGES:
    if (ROOT / im / 'DONE').exists():
        continue
    lg = open(ROOT / 'logs' / f'work_{im}.log', 'a', encoding='utf-8')
    procs[im] = subprocess.Popen([sys.executable, '-u', 'work104b.py', im, str(ROOT)], cwd=str(HERE), stdout=lg, stderr=subprocess.STDOUT)
status['workers'] = {im: p.pid for im, p in procs.items()}; save()
for im, p in procs.items():
    p.wait()
status['workers_done'] = {im: (ROOT / im / 'DONE').exists() for im in C.IMAGES}; save()
if not all(status['workers_done'].values()):
    status['stopped'] = 'worker_failed'; save(); sys.exit(2)
lg = open(ROOT / 'logs' / 'analyze.log', 'a', encoding='utf-8')
subprocess.call([sys.executable, '-u', 'analyze104b.py', str(ROOT), str(EVAL)], cwd=str(HERE), stdout=lg, stderr=subprocess.STDOUT)
status['complete'] = (EVAL / 'index.html').exists()
status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save()
print('CHAIN', 'COMPLETE' if status['complete'] else 'ANALYSIS FAILED', flush=True)
