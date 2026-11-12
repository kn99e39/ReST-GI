"""WL89 configuration. Imports the frozen WL86 / WL87 / WL88 code unchanged (fx88 -> fx87 -> pt86) and injects the WL89 workload family
(fresh moving-sharp candidates, held-out edit, held-out sources) into the shared fixture dictionary.

Material family: the WL88 'wl88' table (box A = glossy mover G with the frozen S1 ER state; box B Lambert). R1 (SAM) keeps the WL88
relation parameters (fx88.P88['relation']) bit-for-bit; WL89 parameters live in protocol/gm1_relation89.json.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gm1_relation88'))
sys.path.insert(0, HERE)
import fx88  # noqa: E402

PT, Q = fx88.PT, fx88.Q
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
FX = PT.FX
P89 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_relation89.json')))
if os.environ.get('WL89_DEV'):                 # dev smoke only: never touch the evaluation candidates / held-out edit
    P89['w_order'] = [P89['dev_workload']]; P89['w2'] = P89['dev_workload']


def inject():
    for name, w in P89['workloads'].items():
        FX[name] = {'T0': w['T0'], 'T1': w['T1'], 'moved': w['moved'], 'light_set': w['light_set'], 'role': w['role']}
    for k, v in P89['light_sets'].items():
        FX['light_sets'][k] = v
    fx88.P88['w2'] = P89['w2']                 # run88.inject_choice maps W2 to the WL89 held-out edit (WL88 file unchanged)


inject()
