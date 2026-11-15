"""WL96 CPU-only check: CK2 incident-bin support at the S1 panel (numpy geometry, unshadowed upper bound; the deployed 16 source points)."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'ecro86'))
import scene86 as S
sc = json.load(open(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
G = S.Geom(sc['geom'])
S1 = np.arange(1088, 1130)
rs = np.random.default_rng(S.P['representation'].get('table_seed', 86001))
us = rs.random((4, 1, S.K), np.float32); usrc = rs.random((2, 1, S.PSRC), np.float32)
L = np.array(sc['lights'][0]['pos'], float)
bins = {}; ring = np.zeros(4); E = np.zeros(64)
for p in S1:
    for j in range(S.PSRC):
        x = G.PO[p] + usrc[0, 0, j] * G.PE1[p] + usrc[1, 0, j] * G.PE2[p]
        d = L - x; dist = np.linalg.norm(d); w = d / dist
        z = w @ G.PN[p]
        if z <= 1e-6:
            continue
        iz = min(int(z * 4), 3); ip = min(int((math.atan2(w @ G.PB[p], w @ G.PT[p]) + math.pi) * 16 / (2 * math.pi)), 15)
        b = iz * 16 + ip
        e = z / dist ** 2
        E[b] += e; ring[iz] += e
print('support bins (incident, unshadowed):', sorted(np.nonzero(E)[0].tolist()))
print('irradiance share by ring:', (ring / ring.sum()).round(4).tolist())
print('bin shares:', {int(b): round(float(E[b] / E.sum()), 4) for b in np.nonzero(E)[0]})
np.save(os.path.join(os.path.dirname(__file__), 'ck2_support_E.npy'), E)
