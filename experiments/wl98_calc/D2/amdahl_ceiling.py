"""WL98 D2 scratch (design input only): zero-cost and measured-cost utility ceilings of ECRO-family learned elements on the
WL96A edit / source lifecycle (N 1,270, one light, 512^2 x 64 spp frame), by which explicit stages a learned element removes.

Inputs are the recorded WL96A wall times (WL96A:132-153; WL97 F0-06 breakdown) and the WL86 per-edit learned costs (WL86
throughput probe: source encoding 5 ms, operator 11 ms). Arithmetic only: no measurement, overhead-dominated prototype
(GPU kernel time 25.5 of 144 ms, F0-06), so the ceilings bound THIS implementation, not the class.
"""
import json, os

EDIT = dict(render=59.5, geometry=30.2, qtable=25.7, relation=18.1 + 5.0, prefix=2.0, tail=3.3)   # sums to 143.8 ~ 144.1
SRC = dict(render=58.5, qtable=24.4, prefix=2.1, tail=3.3)                                       # 88.3 ~ 88.5
TOT_E, TOT_S = 144.1, 88.5
LEARNED_EDIT = 5.0 + 11.0          # WL86: source encoding + generator/application (whole respond stage incl. a CPU cast, J5)
LEARNED_SRC = 5.0 + 0.24           # source encoding + application only (generator cached; WL96A Lambda apply 0.16-0.24 ms)

cases = {
    'resolvent_only (WL96A role)': ['tail'],
    'prefix+tail (learned owns q-table, first order and tail)': ['qtable', 'prefix', 'tail'],
    'prefix+tail+relation (learned also replaces the relation update = learned C)': ['qtable', 'prefix', 'tail', 'relation'],
}
out = {}
for name, own in cases.items():
    r = {}
    for frame in (True, False):
        te = TOT_E - (0 if frame else EDIT['render'])
        ts = TOT_S - (0 if frame else SRC['render'])
        se = sum(EDIT[k] for k in own if k in EDIT)
        ss = sum(SRC[k] for k in own if k in SRC)
        r['frame' if frame else 'no_frame'] = dict(
            edit_zero_cost=round(te / (te - se), 3), edit_measured_learned_cost=round(te / (te - se + LEARNED_EDIT), 3),
            source_zero_cost=round(ts / (ts - ss), 3), source_measured_learned_cost=round(ts / (ts - ss + LEARNED_SRC), 3))
    out[name] = r
    print(name, json.dumps(r))
p = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'wl98_calc', 'D2', 'amdahl_ceiling.json')
with open(os.path.abspath(p), 'w', newline='\n') as fh:
    json.dump(dict(inputs=dict(edit=EDIT, source=SRC, tot_edit=TOT_E, tot_source=TOT_S, learned_edit=LEARNED_EDIT,
                               learned_source=LEARNED_SRC), ceilings=out), fh, indent=1)
