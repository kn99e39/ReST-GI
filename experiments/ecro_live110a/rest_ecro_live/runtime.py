"""Locate and authenticate the ECRO runtime tree (frozen WL86/WL100/WL101 sources, WL110 engine, data and checkpoints).

Search order:
  1. $REST_ECRO_RUNTIME                       (explicit)
  2. <PyInstaller bundle>/runtime              (frozen .app / onedir build)
  3. <this package>/../runtime                 (a staged tree next to the package, see tools/stage_runtime.py)
  4. the repository this package lives in      (development: experiments/ + results/ecro_material100)
The tree keeps the repository layout, because the frozen modules find their data relative to their own files.
"""
import hashlib, json, os, sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
MANIFEST = json.loads((PKG / 'runtime_manifest.json').read_text())


def _is_root(p):
    return p is not None and (p / 'experiments' / 'ecro_dyn101' / 'rt101.py').exists()


def find_root():
    cands = []
    if os.environ.get('REST_ECRO_RUNTIME'):
        cands.append(Path(os.environ['REST_ECRO_RUNTIME']))
    if getattr(sys, '_MEIPASS', None):
        cands.append(Path(sys._MEIPASS) / 'runtime')
    cands.append(PKG.parent / 'runtime')
    cands += [PKG.parents[i] for i in range(2, min(5, len(PKG.parents)))]
    for c in cands:
        if _is_root(c):
            return c.resolve()
    raise FileNotFoundError('ECRO runtime tree not found; tried: ' + ', '.join(map(str, cands)) +
                            '. Stage one with tools/stage_runtime.py or set REST_ECRO_RUNTIME.')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def authenticate(root):
    """Checkpoint / config hashes must equal the WL100 / WL101 record; missing runtime files are reported."""
    out = {'root': str(root), 'checkpoints': {}, 'missing': []}
    for rel, want in MANIFEST['checkpoints'].items():
        p = root / rel
        if not p.exists():
            out['missing'].append(rel); continue
        got = sha256(p)
        out['checkpoints'][rel] = {'sha256': got, 'ok': got == want}
    for rel in MANIFEST['modules'] + MANIFEST['data']:
        if not (root / rel).exists():
            out['missing'].append(rel)
    out['ok'] = not out['missing'] and all(v['ok'] for v in out['checkpoints'].values())
    return out


def add_import_paths(root):
    for d in ('experiments/ecro_dyn101', 'experiments/ecro_live110'):
        p = str(root / d)
        if p not in sys.path:
            sys.path.insert(0, p)
