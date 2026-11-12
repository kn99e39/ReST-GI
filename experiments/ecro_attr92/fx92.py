"""WL92 configuration: imports the frozen WL86 / WL86A / WL87 modules unchanged and adds the prospectively defined condition E4.

E4 is chosen from the never-rendered WL87 candidates E3b / E3c by the geometry / visibility-only rule of the protocol and recorded in
<root>/results/e4_choice.json before any E4 radiance exists. inject() makes it addressable as 'E4_T0' / 'E4_T1' (+ sources LMIX4 / LSPOT4).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro86a', 'gm1_explicit87'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
import fx87  # noqa: E402

PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
FX = PT.FX
P92 = json.load(open(os.path.join(HERE, 'protocol', 'ecro_attr92.json')))
P87 = fx87.P87
E87 = P87['explicit']
SEEDS = ['ecro_s8601', 'ecro_s8602']


def inject(root=None, name=None):
    """Make E4 addressable. name: candidate id; default: the recorded choice in <root>/results/e4_choice.json."""
    if name is None:
        fn = os.path.join(root, 'results', 'e4_choice.json')
        if not os.path.exists(fn):
            return None
        name = json.load(open(fn))['chosen']
        if name is None:
            return None
    c = P87['e3_candidates'][name] if name in P87['e3_candidates'] else P87['e3_dev_candidate'][name]
    FX['E4'] = {'T0': c['T0'], 'T1': c['T1'], 'moved': c['moved'], 'light_set': 'L4', 'role': 'WL92 held-out condition (WL87 candidate ' + name + ')'}
    FX['light_sets']['L4'] = [c['light']]
    for k, v in P92['e4']['sources'].get(name, P92['e4']['sources']['E3b']).items():
        FX['light_sets'][k] = v
    return name


def cfg_ls(tag):
    """'E2_T1' or 'E1_T1_LMIX' -> (geometry, light-set name, edit name, cfg)."""
    p = tag.split('_')
    e, t = p[0], p[1]
    return FX[e][t], (p[2] if len(p) > 2 else FX[e]['light_set']), e, f'{e}_{t}'


def mask_file(R, R86, tag):
    p = tag.split('_')
    name = p[0] + ('_' + p[2] if len(p) > 2 else '')
    for root in (R, R86):
        fn = os.path.join(root, 'results', 'masks', name + '.npz')
        if os.path.exists(fn):
            return fn
    raise FileNotFoundError(name)
