"""WL113 configurations: fixtures F0-F4, cameras, dynamic sequences, D113 training distribution (numpy only).
Reuses WL112 common112 helpers (geometry validity, region masks, camera rule) unchanged."""
import copy, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent / 'ecro_sss112'))
import common112 as C112  # noqa: E402

REPO = C112.REPO
PROTOCOL = HERE / 'protocol' / 'sss113.json'
P = json.loads(PROTOCOL.read_text()) if PROTOCOL.exists() else {}
DEV = os.environ.get('WL113_DEV') == '1'
SO = 7000000 if DEV else 0
def dv(full, dev):
    return dev if DEV else full
jd, sha, digest, masks, RES, EDGES = C112.jd, C112.sha, C112.digest, C112.masks, C112.RES, C112.EDGES
with_sphere, geometry_valid, light_valid, rule_cam = C112.with_sphere, C112.geometry_valid, C112.light_valid, C112.rule_cam
V0_CAM, R_SPHERE = C112.V0_CAM, C112.R_SPHERE


def contact_valid(reg):
    return bool(reg['near'].sum() >= 200 and reg['flank'].sum() >= 1000)


def _fx(test_index, light_index, sphere):
    c = C112.wl100_test(test_index)
    g = copy.deepcopy(c['geom']); g['S'] = {'c': [float(v) for v in sphere]}
    return g, c['lights'][light_index]


FIXTURES = {
    'F0': (0, 0, (0.90, 0.40, 0.40)),        # WL100 test cfg0 / WL101 cfg0: development / regression only
    'F1': (2, 0, (0.80, 0.75, 0.40)),        # WL112 F1 (defined, never evaluated in WL112)
    'F2': (4, 1, (1.00, 0.30, 0.41)),        # new: 1 cm gap, low side light
    'F3': (6, 0, (0.75, 0.45, 0.40)),        # new: contact, high back light
    'F4': (7, 1, (1.10, 0.55, 0.40)),        # new: Stage D only (held out from representation selection)
}


def fixture(name):
    return _fx(*FIXTURES[name])


def views(fix):
    if fix == 'F0':
        return C112.views('F0')
    x, y, _ = FIXTURES[fix][2]
    w2 = rule_cam((x, y), 35, 1.75, 20) if x < 0.95 else rule_cam((x, y), 20, 1.5, 22)   # keep the camera inside the room
    return {'W0': rule_cam((x, y), 0, 2.8, 14), 'W1': rule_cam((x, y), -45, 2.2, 18), 'W2': w2}


def sequences():
    g0, l0 = fixture('F0'); g1, l1 = fixture('F1')
    A = [0.0, 0.002, 0.01, 0.05, 0.01, 0.0, 0.05]
    B = [0.90, 0.80, 0.70, 0.80, 0.90]
    Cg = [0.0, 0.03, 0.0, 0.08, 0.0]
    return {
        'SEQ_A_lift': {'light': l0, 'views': ['V0', 'H2'], 'fixture': 'F0', 'states': [with_sphere(g0, 0.9, 0.4, z) for z in A],
                       'label': [f'gap {1000 * z:g} mm' for z in A], 'revisits': [[0, 5], [3, 6], [2, 4]], 'gaps': A},
        'SEQ_B_slide': {'light': l0, 'views': ['V0', 'H1'], 'fixture': 'F0', 'states': [with_sphere(g0, x, 0.4, 0.0) for x in B],
                        'label': [f'x {x:.2f} m' for x in B], 'revisits': [[0, 4], [1, 3]]},
        'SEQ_C_independent': {'light': l1, 'views': ['W0', 'W2'], 'fixture': 'F1', 'states': [with_sphere(g1, 0.80, 0.75, z) for z in Cg],
                              'label': [f'gap {1000 * z:g} mm' for z in Cg], 'revisits': [[0, 2], [0, 4]], 'gaps': Cg},
    }


def exclusion_states():
    out = [fixture(f)[0] for f in FIXTURES]
    for s in sequences().values():
        out += s['states']
    return out


def train_config(split, index):
    """D112 distribution (WL112 protocol values) with WL113 split seeds."""
    D = {'sphere_x': [0.55, 1.25], 'sphere_y': [0.10, 0.75], 'p_contact': 0.4, 'gap_m': [0.0, 0.12], 'clearance_m': 0.05,
         'exclusion_sphere_m': 0.01, 'exclusion_box_m': 0.05}
    seeds = {'train': 11320100, 'val': 11320200}
    rng = np.random.default_rng([seeds[split] + SO, index])
    nl = 4
    excl = exclusion_states()
    while True:
        A = {'c': [float(rng.uniform(-.35, .05)), float(rng.uniform(.25, .75)), .75], 'h': [.22, .35, .50], 'yaw': float(rng.uniform(0, .4))}
        B = {'c': [float(rng.uniform(1.5, 1.7)), float(rng.uniform(.8, 1.3)), .35], 'h': [.15, .18, .35], 'yaw': float(rng.uniform(0, .3))}
        x = float(rng.uniform(*D['sphere_x'])); y = float(rng.uniform(*D['sphere_y']))
        gap = 0.0 if rng.random() < D['p_contact'] else float(rng.uniform(*D['gap_m']))
        g = {'A': A, 'B': B, 'S': {'c': [x, y, R_SPHERE + gap]}}
        if not geometry_valid(g, D['clearance_m']):
            continue
        if any(np.linalg.norm(np.asarray(g['S']['c']) - np.asarray(e['S']['c'])) < D['exclusion_sphere_m'] and
               all(np.linalg.norm(np.asarray(g[b]['c']) - np.asarray(e[b]['c'])) < D['exclusion_box_m'] for b in 'AB') for e in excl):
            continue
        lights = []
        while len(lights) < nl:
            pos = [float(rng.uniform(.35, 1.5)), float(rng.uniform(.85, 1.5)), float(rng.uniform(1.0, 1.7))]
            if light_valid(pos, g):
                lights.append([{'pos': pos, 'rgb': [float(rng.uniform(20, 32))] * 3}])
        return {'split': split, 'index': index, 'geom': g, 'lights': lights}
