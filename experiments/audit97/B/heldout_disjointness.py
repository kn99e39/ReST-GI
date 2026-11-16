"""WL97 audit track B: disjointness of every later 'held-out' condition from the populations that actually trained each checkpoint.

  python heldout_disjointness.py <population_regen.json> <out_json>
Populations: P86 = all 1,024 WL86 train files (frozen ecro_s8601/8602); P92A = P86 minus results/ecro_own92a/results/exclusion.json
(sum_s*/own_s* of WL92A; init of every WL93A-C / WL94* fine-tune); P94 = the 96 files of results/ecro_mat94/select94.json.
Rule under test: box centre within 0.15 m of the SAME box of the condition, or any light within 0.3 m of any condition light
(the WL86 / WL92A / WL94 rule).
"""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
pop = json.load(open(sys.argv[1]))
tr = pop['train']['items']
P86P = json.load(open(os.path.join(ROOT, 'experiments', 'ecro86', 'protocol', 'ecro86.json')))
FX = P86P['fixture']
P87 = json.load(open(os.path.join(ROOT, 'experiments', 'gm1_explicit87', 'protocol', 'gm1_explicit87.json')))
CK2 = json.load(open(os.path.join(ROOT, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
E5 = [c for c in json.load(open(os.path.join(ROOT, 'results', 'ecro_cf92b', 'results', 'e5_candidates.json')))['geometry_admissible'] if c['k'] == 995][0]
ex92a = {e[0] for e in json.load(open(os.path.join(ROOT, 'results', 'ecro_own92a', 'results', 'exclusion.json')))['excluded']['train']}
sel94 = [r['file'] for r in json.load(open(os.path.join(ROOT, 'results', 'ecro_mat94', 'select94.json')))['splits']['train']['files']]

LS = FX['light_sets']
conds = {
    'E1': {'geoms': [FX['E1']['T0'], FX['E1']['T1']], 'lights': [l['pos'] for k in ('L1', 'LMIX', 'LSPOT') for l in LS[k]]},
    'E2': {'geoms': [FX['E2']['T0'], FX['E2']['T1']], 'lights': [l['pos'] for l in LS['L2']]},
    'E3a': {'geoms': [P87['e3_candidates']['E3a'][t] for t in ('T0', 'T1')], 'lights': [P87['e3_candidates']['E3a']['light']['pos']]},
    'E4(=E3b)': {'geoms': [P87['e3_candidates']['E3b'][t] for t in ('T0', 'T1')],
                 'lights': json.load(open(os.path.join(ROOT, 'results', 'ecro_own92a', 'results', 'exclusion.json')))['E4_light_positions']},
    'E3c': {'geoms': [P87['e3_candidates']['E3c'][t] for t in ('T0', 'T1')], 'lights': [P87['e3_candidates']['E3c']['light']['pos']]},
    'E5(k995)': {'geoms': [E5['T0'], E5['T1']], 'lights': [l['pos'] for k in ('L1', 'L2', 'LMIX', 'LSPOT') for l in LS[k]]},
    'CK2': {'geoms': [CK2['geom']], 'lights': [l['pos'] for l in CK2['lights']]},
}
pops = {'P86_all1024': [it for it in tr], 'P92A_928': [it for it in tr if it['file'] not in ex92a], 'P94_96': [it for it in tr if it['file'] in set(sel94)]}
out = {'rule': 'box centre < 0.15 m of the same box, or any light < 0.3 m of any condition light', 'n': {k: len(v) for k, v in pops.items()}, 'conditions': {}}
for cn, c in conds.items():
    lp = np.array(c['lights'])
    row = {}
    for pn, items in pops.items():
        viol_box, viol_light, dbs, dls, dj = [], [], [], [], []
        for it in items:
            db = min(np.linalg.norm(np.array(it['geom'][b]['c']) - np.array(t[b]['c'])) for t in c['geoms'] for b in 'AB')
            dboth = min(max(np.linalg.norm(np.array(it['geom'][b]['c']) - np.array(t[b]['c'])) for b in 'AB') for t in c['geoms'])
            dl = min(float(np.linalg.norm(lp - np.array(l[0]['pos']), axis=1).min()) for l in it['lights'])
            dbs.append(db); dls.append(dl); dj.append(dboth)
            if db < 0.15:
                viol_box.append([it['file'], round(float(db), 4)])
            if dl < 0.3:
                viol_light.append([it['file'], round(float(dl), 4)])
        row[pn] = {'box_violations': viol_box, 'light_violations': viol_light, 'n_files_violating': len({v[0] for v in viol_box + viol_light}),
                   'min_box_m': float(min(dbs)), 'min_light_m': float(min(dls)), 'min_joint_AB_m (max over boxes, nearest geometry)': float(min(dj))}
    out['conditions'][cn] = row
    print(cn, {pn: (r['n_files_violating'], round(r['min_box_m'], 3), round(r['min_light_m'], 3), round(r['min_joint_AB_m (max over boxes, nearest geometry)'], 3)) for pn, r in row.items()})
json.dump(out, open(sys.argv[2], 'w'), indent=1)
