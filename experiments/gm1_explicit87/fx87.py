"""WL87 configuration: imports the frozen WL86 transport module unchanged and adds the prospectively frozen untouched condition E3.

E3 is chosen from protocol['e3_candidates'] by a geometry / visibility-only rule (mask pixel counts, no radiance) and
recorded in results/e3_choice.json (run87.py e3select) before any E3 radiance exists. inject() puts it into the shared WL86 fixture dictionary so that the
WL86 code (pt86.config, masks86.compute, ecro86.render_cfg) can address it as 'E3_T0' / 'E3_T1'.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
E86 = os.path.join(HERE, '..', 'ecro86')
sys.path.insert(0, E86)
sys.path.insert(0, HERE)
import pt86 as PT  # noqa: E402

mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
FX = PT.FX
P87 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_explicit87.json')))


def candidate(name):
    c = P87['e3_candidates'][name] if name in P87['e3_candidates'] else P87['e3_dev_candidate'][name]
    return {'T0': c['T0'], 'T1': c['T1'], 'moved': c['moved'], 'light_set': 'L3_' + name,
            'role': 'WL87 untouched condition (candidate ' + name + ')'}, c['light']


def inject(root=None, name=None):
    """Make E3 addressable. name: candidate id; default: the recorded choice in <root>/results/e3_choice.json."""
    if name is None:
        fn = os.path.join(root, 'results', 'e3_choice.json')
        if not os.path.exists(fn):
            return None
        name = json.load(open(fn))['chosen']
        if name is None:
            return None
    ex, light = candidate(name)
    FX['E3'] = ex
    FX['light_sets'][ex['light_set']] = [light]
    return name
