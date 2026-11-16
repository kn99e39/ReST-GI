"""WL97 audit C2 (server, CPU): build audit render tables for the WL96A image decision check.  Writes ONLY under argv[1] (= ~/rest_x97/C2/R).
  Fp   = e2 + reference e3_p                      (tail without the resolvent domain -> image of e3_o = hl(F) - hl(Fp))
  Foa  = e2 + reference e3_p + e3_o half a        (e3_o-only reference noise: the noise that XOD-vs-F can actually see)
  Fob  = e2 + reference e3_p + e3_o half b
  F    = e2 + e3 (readout identity check against ~/rest_x96a CK_none_F_X_CK2_T1_r0)
"""
import os, sys
import numpy as np
R = sys.argv[1]
z = np.load(os.path.expanduser('~/rest_x94/results/fields/CKF_none.npz'))
g = lambda k: z[k].astype(np.float64)
m = lambda k: 0.5 * (g(k + '_a') + g(k + '_b'))
E2 = m('e2_p') + m('e2_o')
tab = {'F': E2 + m('e3_p') + m('e3_o'), 'Fp': E2 + m('e3_p'), 'Foa': E2 + m('e3_p') + g('e3_o_a'), 'Fob': E2 + m('e3_p') + g('e3_o_b')}
os.makedirs(os.path.join(R, 'results', 'tables'), exist_ok=True)
os.makedirs(os.path.join(R, 'results', 'lists'), exist_ok=True)
np.savez_compressed(os.path.join(R, 'results', 'tables', 'TC2.npz'), **{k: v.astype(np.float32) for k, v in tab.items()})
with open(os.path.join(R, 'results', 'lists', 'c2.txt'), 'w') as fh:
    for k in ('F', 'Fp', 'Foa', 'Fob'):
        fh.write(f'none results/tables/TC2.npz:{k} 86600 4096 CK_none_C2{k}_X_CK2_T1_r0\n')
print('TABLES OK')
