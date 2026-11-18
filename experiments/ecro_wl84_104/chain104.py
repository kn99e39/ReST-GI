"""WL104 chain (resumable; each stage is skipped when its completion artefact exists). Stops on any gate failure or stage error.

  <venv-cu128 python> chain104.py <results_root> <reviewer_dir>
Order: G0/G1/G4 pre gates -> references (3 images) -> G2/G3 gates -> datasets (3 materials) -> training (3 materials; frozen collapse
retry rule) -> ECRO evaluation (3 images) -> analysis. Status in <results_root>/chain.json, logs in <results_root>/logs/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import common104 as C

ROOT = Path(sys.argv[1]).resolve(); EVAL = Path(sys.argv[2]).resolve()
LOGS = ROOT / 'logs'; LOGS.mkdir(parents=True, exist_ok=True)
PY = sys.executable
HERE = Path(__file__).resolve().parent
ST = ROOT / 'chain.json'
status = json.loads(ST.read_text(encoding='utf-8')) if ST.exists() else {'stages': {}}
status['commit'] = subprocess.check_output(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], text=True).strip()
status['smoke'] = C.SMOKE
status.pop('stopped', None); status.pop('complete', None)
TAGS = sorted({i.rsplit('_L', 1)[0] for i in C.IMAGES})
TRAIN_SEEDS = [10401, 10501, 10601]            # frozen: first seed, then two predeclared retries on collapse


def save():
    C.write_json(ST, status)


def run(name, args, done, env=None, ok=None):
    """Success = the completion artefact `done` exists after the run (and ok(done) if given). Exit codes are recorded but not trusted:
    on this Windows stack the interpreter can abort inside os._exit after a completed stage (WL103)."""
    done = Path(done)
    if done.exists() and (ok is None or ok(done)):
        status['stages'].setdefault(name, {'skipped_existing': True}); save(); return True
    t0 = time.time()
    status['stages'][name] = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'state': 'running'}; save()
    e = dict(os.environ, **(env or {}))
    with open(LOGS / f'{name}.log', 'a', encoding='utf-8') as lg:
        rc = subprocess.call([PY, '-u', *args], cwd=str(HERE), stdout=lg, stderr=subprocess.STDOUT, env=e)
    good = done.exists() and (ok is None or ok(done))
    status['stages'][name].update({'state': 'done' if good else 'failed', 'exit_code': rc, 'seconds': time.time() - t0}); save()
    return good


def passed(f):
    return bool(json.loads(Path(f).read_text(encoding='utf-8')).get('pass'))


def stop(why):
    status['stopped'] = why; status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); save()
    print('CHAIN STOPPED', why, flush=True)
    os._exit(2)


gate_ok = (lambda f: True) if C.SMOKE else passed          # smoke: gates are recorded, not enforced (tiny sample counts)
if not run('gates_pre', ['gates104.py', 'pre', str(ROOT / 'gates_pre.json')], ROOT / 'gates_pre.json', ok=gate_ok):
    stop('gates_pre')
for im in C.IMAGES:
    if not run(f'refs_{im}', ['refs104.py', im, str(ROOT / 'refs' / im)], ROOT / 'refs' / im / 'refs_timing.json'):
        stop(f'refs_{im}')
if not run('gates_refs', ['gates104.py', 'refs', str(ROOT / 'refs'), str(ROOT / 'gates_refs.json')], ROOT / 'gates_refs.json', ok=gate_ok):
    stop('gates_refs')
for tag in TAGS:
    if not run(f'gen_{tag}', ['gen104.py', tag, str(ROOT / 'data' / tag)], ROOT / 'data' / tag / 'gen_test.json'):
        stop(f'gen_{tag}')
for tag in TAGS:
    out = ROOT / 'models' / tag
    if (out / 'training.json').exists():
        continue
    attempts = []
    for seed in TRAIN_SEEDS:
        od = ROOT / 'models' / f'{tag}_seed{seed}'
        args = ['-c', f'import sys; sys.path.insert(0, r"{C.EXP / "ecro_lr3_102"}"); sys.path.insert(0, r"{C.EXP / "ecro86"}"); '
                f'import train102 as T; T.main(r"{ROOT / "data" / tag}", r"{od}"' + (', 300)' if C.SMOKE else ')')]
        if not run(f'train_{tag}_seed{seed}', args, od / 'training.json', env={'WL102_SEED': str(seed), 'WL102_AREA_FLOOR': '0'}):
            stop(f'train_{tag}_seed{seed}')
        tr = json.loads((od / 'training.json').read_text(encoding='utf-8'))
        collapsed = tr['validation']['whole_rel_l1'] >= 0.95          # frozen collapse rule (zero / near-zero output)
        attempts.append({'seed': seed, 'val_whole_rel_l1': tr['validation']['whole_rel_l1'], 'collapsed': collapsed})
        if not collapsed:
            out.mkdir(parents=True, exist_ok=True)
            for f in ('model.pt', 'training.json'):
                (out / f).write_bytes((od / f).read_bytes())
            break
    status.setdefault('training_attempts', {})[tag] = attempts; save()
    if not (out / 'training.json').exists():
        stop(f'train_{tag}_all_seeds_collapsed')
for im in C.IMAGES:
    tag = im.rsplit('_L', 1)[0]
    if not run(f'eval_{im}', ['eval104.py', im, str(ROOT / 'data' / tag), str(ROOT / 'models' / tag), str(ROOT / 'eval' / im)],
               ROOT / 'eval' / im / 'eval.json'):
        stop(f'eval_{im}')
if not run('analyze', ['analyze104.py', str(ROOT), str(EVAL)], EVAL / 'index.html'):
    stop('analyze')
status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); status['complete'] = True; save()
print('CHAIN COMPLETE', flush=True)
