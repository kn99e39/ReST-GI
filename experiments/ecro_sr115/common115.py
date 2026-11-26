"""WL115 configurations (numpy only).

Evidence classes (fixed before any WL115 evidence stream):
  SELECTION  - WL113 fixtures F0-F3 and sequences SEQ_A / SEQ_B / SEQ_C (all their WL114 views): the ONLY configurations used for
               Stage A attribution and Stage B representation selection / calibration.
  HELD_SEEN  - F4, N1-N4, SEQ_D: evaluated in WL114 (S4_F, B1, B2) but never used for any WL115 selection, descriptor or calibration
               decision (regression / held-out evidence).
  FRESH      - N5-N8 and SEQ_E: new independent configurations drawn here (boxes / lights from the D112 sampling rule with a new seed
               family; sphere support positions and gaps fixed by the stress design below) before any WL115 evidence; never used for
               selection, descriptors, calibration or training-target construction.
WL114 common114 / WL113 common113 are imported unchanged.
"""
import copy, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_fd114'))
import common114 as C114  # noqa: E402

C113, C112 = C114.C113, C114.C112
REPO = C114.REPO
PROTOCOL = HERE / 'protocol' / 'sr115.json'
P = json.loads(PROTOCOL.read_text()) if PROTOCOL.exists() else {}
DEV = os.environ.get('WL115_DEV') == '1'
SO = 7000000 if DEV else 0
def dv(full, dev):
    return dev if DEV else full
jd, sha, masks, RES = C114.jd, C114.sha, C114.masks, C114.RES
with_sphere, geometry_valid, light_valid, rule_cam = C114.with_sphere, C114.geometry_valid, C114.light_valid, C114.rule_cam
contact_valid, R_SPHERE = C114.contact_valid, C114.R_SPHERE
WL114_ROOT = REPO / 'results' / 'ecro_fd114' / 'primary'

SELECTION_FIX = ('F0', 'F1', 'F2', 'F3')
SELECTION_SEQ = ('SEQ_A_lift', 'SEQ_B_slide', 'SEQ_C_independent')
HELD_FIX = ('F4', 'N1', 'N2', 'N3', 'N4')
HELD_SEQ = ('SEQ_D_independent',)


def gdigest(g):
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ fresh independent configurations
FRESH_SEED = 11515000
# stress design (support xy, gap) fixed before any evidence: two contacts (one support point on a floor-patch corner, one at a patch
# centre), one small gap, one lifted state
FRESH_SUPPORT = {'N5': ((1.00, 0.50), 0.0), 'N6': ((0.625, 0.375), 0.0), 'N7': ((1.20, 0.30), 0.008), 'N8': ((0.80, 0.55), 0.045)}


def _fresh_scene(tag_index):
    """Boxes and one light from the D112 sampling rule (WL113 train_config ranges) with a new seed family; the sphere is placed
    afterwards by the stress design. Returns (geometry without sphere position check, light cfg)."""
    rng = np.random.default_rng([FRESH_SEED + SO, tag_index])
    while True:
        A = {'c': [float(rng.uniform(-.35, .05)), float(rng.uniform(.25, .75)), .75], 'h': [.22, .35, .50], 'yaw': float(rng.uniform(0, .4))}
        B = {'c': [float(rng.uniform(1.5, 1.7)), float(rng.uniform(.8, 1.3)), .35], 'h': [.15, .18, .35], 'yaw': float(rng.uniform(0, .3))}
        pos = [float(rng.uniform(.35, 1.5)), float(rng.uniform(.85, 1.5)), float(rng.uniform(1.0, 1.7))]
        yield {'A': A, 'B': B}, [{'pos': pos, 'rgb': [float(rng.uniform(20, 32))] * 3}]


def fresh_fixture(name):
    (x, y), gap = FRESH_SUPPORT[name]
    gen = _fresh_scene(int(name[1:]))
    for g0, l in gen:
        g = dict(copy.deepcopy(g0), S={'c': [x, y, R_SPHERE + gap]})
        if geometry_valid(g, 0.05) and light_valid(l[0]['pos'], g):
            return g, l
    raise RuntimeError(name)


def fresh_views(name):
    (x, y), _ = FRESH_SUPPORT[name]
    w2 = rule_cam((x, y), 35, 1.75, 20) if x < 0.95 else rule_cam((x, y), 20, 1.5, 22)
    return {'W0': rule_cam((x, y), 0, 2.8, 14), 'W2': w2}


def seq_E():
    """New independent dynamic sequence: contact slide across the floor-patch boundary x = 0.75, contact removal, small and large
    separations, contact re-formation (revisit) and a large slide back to the first state (revisit)."""
    gen = _fresh_scene(9)
    path = [((0.62, 0.30), 0.0), ((0.70, 0.30), 0.0), ((0.78, 0.30), 0.0), ((0.78, 0.30), 0.005), ((0.78, 0.30), 0.020),
            ((0.78, 0.30), 0.070), ((0.78, 0.30), 0.0), ((0.62, 0.30), 0.0)]
    for g0, l in gen:
        states = [dict(copy.deepcopy(g0), S={'c': [x, y, R_SPHERE + z]}) for (x, y), z in path]
        if all(geometry_valid(g, 0.05) for g in states) and light_valid(l[0]['pos'], states[0]) and all(light_valid(l[0]['pos'], g) for g in states):
            break
    xy = (0.70, 0.30)
    lab = [f'xy ({x:.2f},{y:.2f}) gap {1000 * z:g} mm' for (x, y), z in path]
    return {'light': l, 'views': ['W0', 'W2'], 'fixture': None, 'xy': xy, 'states': states, 'label': lab,
            'revisits': [[2, 6], [0, 7]], 'gaps': None, 'track_states': [2, 3, 4, 5], 'track_gaps': [0.0, 0.005, 0.020, 0.070]}


def sequences():
    s = C114.sequences()
    s['SEQ_E_fresh'] = seq_E()
    return s


def seq_views(seq):
    if seq.get('fixture'):
        return C113.views(seq['fixture'])
    x, y = seq['xy']
    w2 = rule_cam((x, y), 35, 1.75, 20) if x < 0.95 else rule_cam((x, y), 20, 1.5, 22)
    return {'W0': rule_cam((x, y), 0, 2.8, 14), 'W1': rule_cam((x, y), -45, 2.2, 18), 'W2': w2}


def fixture(name):
    return fresh_fixture(name) if name in FRESH_SUPPORT else C114.fixture(name)


def views(name):
    return fresh_views(name) if name in FRESH_SUPPORT else C114.views(name)


def selection_geometries():
    """Unique geometries (digest -> (key, geometry, light)) of the selection set."""
    out = {}
    for f in SELECTION_FIX:
        g, l = C114.fixture(f); out.setdefault(gdigest(g), (f, g, l))
    sq = C114.sequences()
    for sname in SELECTION_SEQ:
        q = sq[sname]
        for t, g in enumerate(q['states']):
            out.setdefault(gdigest(g), (f'{sname}_{t}', g, q['light']))
    return out


def fresh_eval_states():
    out = [fresh_fixture(n)[0] for n in FRESH_SUPPORT]
    out += seq_E()['states']
    return out


def isolation_check(train_cfgs):
    """Minimum distance (sphere centre, boxes) between every fresh evaluation state and the given training configurations."""
    res = []
    for g in fresh_eval_states():
        best = 1e9
        for cfg in train_cfgs:
            t = cfg['geom']
            d = np.linalg.norm(np.asarray(g['S']['c']) - np.asarray(t['S']['c'])) + sum(np.linalg.norm(np.asarray(g[b]['c']) - np.asarray(t[b]['c'])) for b in 'AB')
            best = min(best, float(d))
        res.append(best)
    return res
