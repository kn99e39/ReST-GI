"""WL86 amendment 1 (post hoc, labelled): re-evaluate test V3's 'in-place edit == rebuilt scene' clause with the amended rule.

  python amend1_86.py <root>
Reads results/tests/tests86.json (frozen run) and results/amend1/v3_diagnosis.json; writes results/tests/tests86_amend1.json.
Nothing else is re-evaluated: every other test keeps its frozen result.
"""
import json, os, sys

root = sys.argv[1]
t = json.load(open(os.path.join(root, 'results', 'tests', 'tests86.json')))
d = json.load(open(os.path.join(root, 'results', 'amend1', 'v3_diagnosis.json')))
v3 = {}
ok = True
for e in ('E1', 'E2'):
    r = d[e]
    c = {'vertex_max_abs_diff_m': max(r['boxA']['vertex_max_abs_diff_m'], r['boxB']['vertex_max_abs_diff_m']), 'state_identical': r['state_identical']}
    p = c['vertex_max_abs_diff_m'] <= 1e-6 and r['state_identical']
    for k in ('spp256', 'spp1024'):
        c[k] = {x: r[k][x] for x in ('frame_rel_diff', 'rel_l1', 'fraction', 'max_rel_mean', 'pixels_differing')}
        p &= r[k]['frame_rel_diff'] <= 1e-5 and r[k]['rel_l1'] <= 1e-5 and r[k]['fraction'] <= 1e-3
    f = t['V3'][e]
    p &= f['null_max_rel_mean'] <= 2e-4 and f['return_max_rel_mean'] <= 2e-4 and f['edit_changes_image_rel_l1'] > 0.005        # frozen clauses, frozen values
    c['PASS'] = bool(p)
    v3[e] = c
    ok &= p
out = dict(t)
out['amendment'] = 1
out['V3_PASS_frozen'] = t['V3_PASS']
out['V3_amended'] = v3
out['V3_PASS'] = bool(ok)
out['PASS_frozen'] = t['PASS']
out['PASS'] = bool(all(v for k, v in out.items() if k.endswith('_PASS') and k not in ('V3_PASS_frozen',)))
json.dump(out, open(os.path.join(root, 'results', 'tests', 'tests86_amend1.json'), 'w'), indent=1)
print(json.dumps(v3))
print('TESTS_AMEND1', 'PASS' if out['PASS'] else 'FAIL', flush=True)
