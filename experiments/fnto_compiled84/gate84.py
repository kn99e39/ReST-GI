"""WL84 verdict (protocol outcomes).

  python gate84.py audit <out>     -> REPRODUCED | NOT_REPRODUCED
  python gate84.py verdict <out>   -> writes verdict84.json, prints the outcome
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P84 = json.load(open(os.path.join(HERE, 'protocol', 'compiled84.json')))
IMAGES = P84['images']


def t_at(curve, e):
    """GPU time (ms) at which the layered-reference ladder reaches image error e: log-log interpolation between ladder points; beyond the
    ladder, extrapolation from the largest-spp point with err ~ t^-1/2 (Monte Carlo rate)."""
    t = np.array([p['gpu_ms'] for p in curve]); y = np.array([p['err'] for p in curve])
    if e >= y[0]:
        return float(t[0] * (y[0] / e) ** 2), 'below_ladder'
    for i in range(len(y) - 1):
        if y[i] >= e >= y[i + 1]:
            s = math.log(y[i + 1] / y[i]) / math.log(t[i + 1] / t[i])
            return float(t[i] * (e / y[i]) ** (1 / s)), 'interpolated'
    return float(t[-1] * (y[-1] / e) ** 2), 'extrapolated'


def verdict(out):
    au = json.load(open(os.path.join(out, 'audit', 'audit84.json')))
    ev = json.load(open(os.path.join(out, 'eval84.json')))['rows']
    ec = json.load(open(os.path.join(out, 'econ', 'econ84.json')))
    contention = bool(ec['foreign_gpu_processes_before'] or ec['foreign_gpu_processes_after'])
    econ = {}
    for im in IMAGES:
        g = ec['images'][im]
        ref64 = g['REF']['per_64spp_median']['gpu_ms']; er64 = g['ER_L']['per_64spp_median']['gpu_ms']
        e_er = [p for p in g['curves']['ER_L'] if p['spp'] == 64][0]
        t_ref, how = t_at(g['curves']['REF'], e_er['err'])
        econ[im] = {'gpu_ms_per64_REF': ref64, 'gpu_ms_per64_ER_L': er64, 'gpu_ratio_REF_over_ER': ref64 / er64,
                    'wall_s_per64_REF': g['REF']['per_64spp_median']['wall_s'], 'wall_s_per64_ER_L': g['ER_L']['per_64spp_median']['wall_s'],
                    'ER_L_err_at_64': e_er['err'], 'REF_gpu_ms_to_reach_it': t_ref, 'how': how, 'time_to_quality_ratio_gpu': t_ref / e_er['gpu_ms'],
                    'i_per_spp': ref64 / er64 >= 1.10, 'ii_time_to_quality': t_ref / e_er['gpu_ms'] >= 1.10}
    favourable = all(v['i_per_spp'] and v['ii_time_to_quality'] for v in econ.values())
    vt = {im: ev[im]['ER_L_verdict'] for im in IMAGES}
    if not au['REPRODUCED'] or contention:
        outcome = 'E'
    elif vt['test83_iid_1_L2'] == 'INDETERMINATE':
        outcome = 'C'
    elif any(v == 'FAIL' for v in vt.values()):
        outcome = 'B'
    elif all(v == 'PASS' for v in vt.values()):
        outcome = 'A' if favourable else 'D'
    else:
        outcome = 'C'
    res = {'reproduced': au['REPRODUCED'], 'contention': contention, 'ER_L_per_image': vt,
           'E_L_per_image': {im: ev[im]['E_L_verdict'] for im in IMAGES}, 'economics': econ, 'ECONOMICS_FAVOURABLE': favourable, 'OUTCOME': outcome}
    json.dump(res, open(os.path.join(out, 'verdict84.json'), 'w'), indent=1)
    print(outcome)


if __name__ == '__main__':
    if sys.argv[1] == 'audit':
        print('REPRODUCED' if json.load(open(os.path.join(sys.argv[2], 'audit', 'audit84.json')))['REPRODUCED'] else 'NOT_REPRODUCED')
    else:
        verdict(sys.argv[2])
