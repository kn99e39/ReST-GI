"""Rank of the continuous-incidence -> receiver-patch transfer of one S1 sub-cell (unoccluded, rect patches of the CK2 geometry).
T[w_i, r] = A_c * sum_{x in r} f_ER(c->x local, w_i local) cos_c cos_x / d^2 dA_x   (unit incident radiance-times-solid-angle scale)."""
import sys, json, math
import numpy as np
sys.path.insert(0, r'c:/Projects/ReST-GI/experiments/ecro86')
import scene86 as S
from er_np import ER
sc = json.load(open(r'c:/Projects/ReST-GI/experiments/ecro_local93d/addendum_checker/scene_ck2.json'))
G = S.Geom(sc['geom'])
mats = np.load(r'c:/Projects/ReST-GI/experiments/ecro86/data/mats86.npz')['states']
rect = np.nonzero((G.PSID != S.SID_SPHERE) & (G.PMAT != 1))[0]
m = 3
g = (np.arange(m) + .5) / m; A, B = np.meshgrid(g, g, indexing='ij'); A = A.ravel(); B = B.ravel()
X = G.PO[rect][:, None] + A[None, :, None] * G.PE1[rect][:, None] + B[None, :, None] * G.PE2[rect][:, None]   # [R, m2, 3]
NX = G.PN[rect]; dA = (G.PAREA[rect] / (m * m))
out = {}
for mat in (1, 2, 0):
    er = ER(mats[mat])
    for name, c in (('corner', np.array([-1.5, -0.275, 1.575])), ('centre', np.array([-1.5, 0.3, 1.0]))):
        if mat != 1 and name == 'corner':
            continue
        n = np.array([1.0, 0, 0]); t = np.array([0, 1.0, 0]); b = np.array([0, 0, 1.0])
        nzz, nph = 48, 96
        zz = (np.arange(nzz) + .5) / nzz; ph = (np.arange(nph) + .5) / nph * 2 * math.pi - math.pi
        Z, P = np.meshgrid(zz, ph, indexing='ij'); Z = Z.ravel(); P = P.ravel(); r_ = np.sqrt(1 - Z * Z)
        wi_l = np.stack([r_ * np.cos(P), r_ * np.sin(P), Z], -1)                     # incident (light) directions, local
        d = X - c; d2 = (d * d).sum(-1); v = d / np.sqrt(d2)[..., None]
        cc = v @ n; cx = -(v * NX[:, None]).sum(-1)
        ok = (cc > 1e-6) & (cx > 1e-6)
        vl = np.stack([v @ t, v @ b, v @ n], -1)                                     # outgoing (viewer) local
        geo = np.where(ok, cc * cx / d2, 0) * dA[:, None]                           # [R, m2]
        vflat = vl.reshape(-1, 3); gflat = geo.reshape(-1)
        keep = gflat > 0; vk = vflat[keep]; gk = gflat[keep]
        T = np.zeros((len(Z), len(rect)))
        ridx = np.repeat(np.arange(len(rect)), m * m)[keep]
        for i0 in range(0, len(Z), 64):
            wi = wi_l[i0:i0 + 64]
            f = er.eval(np.broadcast_to(vk[None], (len(wi),) + vk.shape), np.broadcast_to(wi[:, None], (len(wi), len(vk), 3))).sum(-1)
            val = f * gk[None]
            for j in range(len(wi)):
                T[i0 + j] = np.bincount(ridx, weights=val[j], minlength=len(rect))
        # weight incidence rows by cos(theta_i) solid-angle measure (uniform in z already = cos-weighted area measure / equal-solid-angle grid)
        U_, s, Vt = np.linalg.svd(T, full_matrices=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        res = {'rows_incidence': int(len(Z)), 'receiver_patches': int(len(rect)),
               'rank_energy90': int(np.searchsorted(e, .9) + 1), 'rank_energy99': int(np.searchsorted(e, .99) + 1)}
        for rk in (16, 32, 64, 128, 256):
            Tr = (U_[:, :rk] * s[:rk]) @ Vt[:rk]
            res[f'rel_l1_rank{rk}'] = float(np.abs(Tr - T).sum() / np.abs(T).sum())
            Tc = np.clip(Tr, 0, None); res[f'rel_l1_rank{rk}_clip0'] = float(np.abs(Tc - T).sum() / np.abs(T).sum())
        # grazing rows only (cos_i < .25: ring 0)
        gr = Z < .25
        for rk in (16, 64):
            Tr = (U_[:, :rk] * s[:rk]) @ Vt[:rk]
            res[f'rel_l1_rank{rk}_ring0rows'] = float(np.abs(Tr[gr] - T[gr]).sum() / np.abs(T[gr]).sum())
        # per-row receiver support: number of receiver patches carrying 90 % of a row's flux (median over rows)
        srt = -np.sort(-T, 1); cs = np.cumsum(srt, 1) / np.maximum(srt.sum(1, keepdims=True), 1e-30)
        res['receivers_for_90pct_row_flux_median'] = float(np.median((cs < .9).sum(1) + 1))
        out[f'mat{mat}_{name}'] = res
        print(f'mat{mat}_{name}', json.dumps(res), flush=True)
json.dump(out, open('rank_transfer.json', 'w'), indent=1)
