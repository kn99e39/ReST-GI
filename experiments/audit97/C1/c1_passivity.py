"""WL97 C1: directional albedo of the learned S1 local operator A = sum_r a c^T (CK2 geometry) per incident bin, in the SAME convention as
the WL94D T_ENERGY check of the M64 teacher (s94.py:179-181): alb[bi, c] = sum_bo M[bi, bo, c] * zb[bo] * dOmega, zb = true ring cosine.
numpy only, from the c1_sender_dump.py output.

  python c1_passivity.py <SND_DIR> <out.json>
"""
import json, os, sys
import numpy as np

SND, OUTFN = sys.argv[1:3]
NB = 64
ZB = ((np.arange(NB) // 16) + 0.5) / 4
DOM = 2 * np.pi / NB
M1 = np.load(os.path.join(SND, 'M64_S1.npy')).astype(np.float64)
g = np.load(os.path.join(SND, 'ck2_geom.npz'))
s = g['s'].astype(np.float64)[1088:1130]                    # CK2 incident field on the panel [42, NB, 3]
lit = s.sum((0, 2)) > 0
albM = np.einsum('ioc,o->ic', M1, ZB) * DOM
out = {'M64_S1': {'albedo_max': float(albM.max()), 'albedo_mean': float(albM.mean()), 'albedo_at_ck2_lit_bins': albM[lit].tolist()},
       'ck2_lit_incident_bins': np.nonzero(lit)[0].tolist(), 'models': {}}
for fn in sorted(os.listdir(SND)):
    if not fn.startswith('SND_'):
        continue
    tag = fn[4:-4]
    A = np.load(os.path.join(SND, fn))['ck2|A'].astype(np.float64)          # [42, bi, bo, 3]
    alb = np.einsum('nioc,o->nic', A, ZB) * DOM
    ring = np.arange(NB) // 16
    out['models'][tag] = {'albedo_max': float(alb.max()), 'albedo_mean': float(alb.mean()), 'share_gt_1': float((alb > 1).mean()), 'share_gt_1.05': float((alb > 1.05).mean()),
                          'albedo_max_by_incident_ring': [float(alb[:, ring == r].max()) for r in range(4)],
                          'albedo_mean_by_incident_ring': [float(alb[:, ring == r].mean()) for r in range(4)],
                          'M64_mean_by_incident_ring': [float(albM[ring == r].mean()) for r in range(4)],
                          'albedo_at_ck2_lit_bins_mean_over_patches': alb[:, lit].mean(0).tolist(),
                          'albedo_at_ck2_lit_bins_max_over_patches': alb[:, lit].max(0).tolist(),
                          'min_entry': float(A.min())}
    m = out['models'][tag]
    print('%-10s albedo max %.3f mean %.3f  share>1 %.4f  max by ring %s  mean by ring %s' % (tag, m['albedo_max'], m['albedo_mean'], m['share_gt_1'],
          ' '.join('%.2f' % x for x in m['albedo_max_by_incident_ring']), ' '.join('%.2f' % x for x in m['albedo_mean_by_incident_ring'])), flush=True)
print('M64 S1 albedo max %.3f mean %.3f by ring %s ; CK2 lit bins %s albedo %s' % (albM.max(), albM.mean(), ' '.join('%.2f' % albM[np.arange(NB) // 16 == r].mean() for r in range(4)),
      out['ck2_lit_incident_bins'], np.round(albM[lit], 3).tolist()))
json.dump(out, open(OUTFN, 'w'), indent=1)
