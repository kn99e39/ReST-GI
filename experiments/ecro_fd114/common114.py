"""WL114 configurations (numpy only): WL113 fixtures / sequences (regression, unchanged), new independent fixtures N1-N4 and the new
dynamic sequence SEQ_D, Stage A tail-qualification views, seeds, paths. WL113 common113 is imported unchanged."""
import copy, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common113 as C113  # noqa: E402

C112 = C113.C112
REPO = C113.REPO
PROTOCOL = HERE / 'protocol' / 'fd114.json'
P = json.loads(PROTOCOL.read_text()) if PROTOCOL.exists() else {}
DEV = os.environ.get('WL114_DEV') == '1'
SO = 7000000 if DEV else 0
def dv(full, dev):
    return dev if DEV else full
jd, sha, digest, masks, RES = C113.jd, C113.sha, C113.digest, C113.masks, C113.RES
with_sphere, geometry_valid, light_valid, rule_cam = C113.with_sphere, C113.geometry_valid, C113.light_valid, C113.rule_cam
contact_valid, V0_CAM, R_SPHERE = C113.contact_valid, C113.V0_CAM, C113.R_SPHERE
FIXTURES_113 = C113.FIXTURES
# new independent fixtures (WL100 SSS test configurations / lights never used as WL112-WL113 fixtures; sphere placed by WL114)
NEW = {
    'N1': (1, 0, (0.70, 0.55, 0.40)),        # contact
    'N2': (3, 1, (1.05, 0.65, 0.40)),        # contact
    'N3': (5, 0, (0.85, 0.35, 0.42)),        # lifted 2 cm
    'N4': (1, 1, (1.15, 0.45, 0.46)),        # lifted 6 cm
}
ALL_FIX = dict(FIXTURES_113, **NEW)


def fixture(name):
    return C113._fx(*ALL_FIX[name])


def views(fix):
    if fix in FIXTURES_113:
        return C113.views(fix)
    x, y, _ = ALL_FIX[fix][2]
    w2 = rule_cam((x, y), 35, 1.75, 20) if x < 0.95 else rule_cam((x, y), 20, 1.5, 22)
    return {'W0': rule_cam((x, y), 0, 2.8, 14), 'W1': rule_cam((x, y), -45, 2.2, 18), 'W2': w2}


NEW_VIEWS = ('W0', 'W2')                     # Stage D views of N1-N4


def sequences():
    s = C113.sequences()
    gD, lD = C113._fx(5, 1, (0.95, 0.60, 0.40))
    Dg = [0.0, 0.015, 0.0, 0.04, 0.10, 0.015, 0.0]
    s['SEQ_D_independent'] = {'light': lD, 'views': ['W0', 'W1'], 'fixture': None, 'xy': (0.95, 0.60),
                              'states': [with_sphere(gD, 0.95, 0.60, z) for z in Dg], 'label': [f'gap {1000 * z:g} mm' for z in Dg],
                              'revisits': [[0, 2], [0, 6], [1, 5]], 'gaps': Dg}
    return s


def seq_views(seq):
    if seq.get('fixture'):
        return C113.views(seq['fixture'])
    x, y = seq['xy']
    w2 = rule_cam((x, y), 35, 1.75, 20) if x < 0.95 else rule_cam((x, y), 20, 1.5, 22)
    return {'W0': rule_cam((x, y), 0, 2.8, 14), 'W1': rule_cam((x, y), -45, 2.2, 18), 'W2': w2}


def stageA_views():
    """Stage A tail-qualification views: (tag, geometry, light cfg, camera)."""
    out = []
    for f, v in (('F0', 'V0'), ('F1', 'W0'), ('F2', 'W0'), ('F3', 'W0'), ('F4', 'W0')):
        g, l = fixture(f); out.append((f'{f}_{v}', g, l, views(f)[v]))
    s = sequences()
    for sname, st, v in (('SEQ_A_lift', 3, 'V0'), ('SEQ_C_independent', 1, 'W0'), ('SEQ_C_independent', 3, 'W0')):
        q = s[sname]; out.append((f'{sname}_{st}_{v}', q['states'][st], q['light'], seq_views(q)[v]))
    for f in NEW:
        g, l = fixture(f); out.append((f'{f}_W0', g, l, views(f)['W0']))
    return out


def stageA_constructions():
    """Construction-side tail qualification (probe suffix): fixtures / states."""
    s = sequences()
    return [('F0',) + fixture('F0'), ('F1',) + fixture('F1'), ('SEQ_C_3', s['SEQ_C_independent']['states'][3], s['SEQ_C_independent']['light']),
            ('N3',) + fixture('N3')]


def eval_states():
    """Every evaluation geometry (for training-set isolation checks)."""
    out = [fixture(f)[0] for f in ALL_FIX]
    for q in sequences().values():
        out += q['states']
    return out


train_config = C113.train_config            # D113 configurations (identical seeds) -> D114
