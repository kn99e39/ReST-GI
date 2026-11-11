"""WL88 configuration: the frozen WL86 transport module (pt86) and WL87 explicit code (quad87 / xport87) imported unchanged, plus the
WL88 glossy-mover workload family.

Material table of the WL88 family (frozen ER states, bit-identical copies, no new material):
  0 B floor, 1 S1 panel, 2 S2 sphere, 3 Lambert walls / ceiling, 4 = box A = the moving glossy box G with the S1 ER state (sharp),
  5 Lambert box B.  The 'P' physical reference uses the same layered descriptor as S1 for G.
The WL86 / WL87 historical cases (E1 / E2) keep their own material table: use_family('wl86') restores it.

Workloads (protocol['workloads']) are injected into the shared WL86 fixture dictionary so that the WL86 code (pt86.config, Ctx, render)
can address them as '<name>_T0' / '<name>_T1'.
"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gm1_explicit87'))
sys.path.insert(0, HERE)
import fx87  # noqa: E402
import quad87 as Q  # noqa: E402

PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
FX = PT.FX
P88 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_relation88.json')))
if os.environ.get('WL88_DEV'):                 # dev smoke only: never touch the evaluation candidates / held-out edit
    P88['w_order'] = [P88['dev_workload']]; P88['w2'] = P88['dev_workload']
G_MAT = 4                      # material id (and shape SID_BOXA) of the glossy mover
SHARP_IDS = (1, 4)             # S1 surfaces (panel, G) used for the geometry-only mirror footprints

_z = np.load(os.path.join(fx87.E86, 'data', 'mats86.npz'))
_ORIG = {k: _z[k].copy() for k in ('states', 'desc', 'lambert', 'albedo')}
_FAM = {'wl86': dict(_ORIG)}
_s = _ORIG['states'].copy(); _s[4] = _s[1]
_d = _ORIG['desc'].copy(); _d[4] = _d[1]
_l = _ORIG['lambert'].copy(); _l[4] = 0
_a = _ORIG['albedo'].copy(); _a[4] = 0.0
_FAM['wl88'] = {'states': _s, 'desc': _d, 'lambert': _l, 'albedo': _a}
_CLASS = {'wl86': list(S.MAT_CLASS), 'wl88': [2, 0, 1, 2, 0, 2]}
FAMILY = {'name': None}
_MATS86_CLASS = PT.Mats


class Mats88(_MATS86_CLASS):
    def __init__(self):
        m = _FAM[FAMILY['name']]
        self.states = m['states']; self.desc = m['desc']; self.lambert = m['lambert']; self.albedo = m['albedo']
        self.desc_cols = [PT.fl(self.desc[:, j]) for j in range(10)]
        self.lam_f = PT.fl(self.lambert); self.alb_f = PT.fl(self.albedo)
        self.cls = PT.U(np.array(_CLASS[FAMILY['name']], np.uint32))


def _mats():
    m = _FAM[FAMILY['name']]
    return m['states'], m['lambert'].astype(bool), m['albedo'].astype(np.float64)


def use_family(name):
    """'wl88' (glossy mover G = S1 on box A) or 'wl86' (historical E1 / E2 material table). Affects Ctx created afterwards."""
    FAMILY['name'] = name
    PT.Mats = Mats88
    Q.mats = _mats
    S.MAT_CLASS[:] = _CLASS[name]


def family_of(cfg):
    return 'wl86' if cfg.split('_')[0] in ('E1', 'E2', 'E3') else 'wl88'


def inject():
    for name, w in P88['workloads'].items():
        FX[name] = {'T0': w['T0'], 'T1': w['T1'], 'moved': w['moved'], 'light_set': w['light_set'], 'role': w['role']}
        if 'T2' in w:
            FX[name]['T2'] = w['T2']
    for k, v in P88['light_sets'].items():
        FX['light_sets'][k] = v


def setup(cfg):
    """Select the material family of a configuration name ('W1_T1', 'E2_T1', ...)."""
    use_family(family_of(cfg))


inject()
use_family('wl88')
