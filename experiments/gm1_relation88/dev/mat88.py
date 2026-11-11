"""WL88 dev (material-only, no transport): angular peakedness of the frozen ER materials.

Pi_m(w) = max_bo fbar_m(w)[bo] / (a_m(w) / pi), a_m(w) = sum_bo fbar_m(w)[bo] * DOMEGA * zc_bo  (directional albedo, bin-centre cosine)
Lambert: Pi = 1. Evaluated at the 64 bin-centre incident directions with the WL87 fbar estimator (frozen ER, MIS mixture).
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gm1_explicit87'))
import fx87  # noqa: E402
import quad87 as Q  # noqa: E402
S = fx87.S
NB = S.NB
st, lam, alb = Q.mats()
dirs = S.bin_dirs()
zc = (np.arange(NB) // S.NPHI + 0.5) / S.NZ
out = {}
for m, name in [(0, 'B'), (1, 'S1'), (2, 'S2')]:
    fb = Q.fbar(st[m], dirs, 65536, 88000 + m).mean(-1)          # luminance [64 in, 64 out]
    a = (fb * S.DOMEGA * zc[None]).sum(1)
    pi_w = fb.max(1) / np.maximum(a / math.pi, 1e-12)
    out[name] = {'Pi_mean': float(pi_w.mean()), 'Pi_median': float(np.median(pi_w)), 'Pi_min': float(pi_w.min()), 'Pi_max': float(pi_w.max()),
                 'albedo_mean': float(a.mean()), 'Pi_by_ring': [float(pi_w[r * S.NPHI:(r + 1) * S.NPHI].mean()) for r in range(S.NZ)]}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(os.path.join(sys.argv[1] if len(sys.argv) > 1 else '.', 'mat88.json'), 'w'), indent=1)
