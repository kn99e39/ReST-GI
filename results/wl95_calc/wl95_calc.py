"""WL95 Stage B offline CPU calculation (read-only inputs; writes only ~/wl95_calc/).
Matched-norm representability bounds for the M64 local operator (S1 panel mat 1; contrast B mat 0, S2 mat 2).
Metrics copied from experiments/ecro_s94d/s94.py:sender (operator, rel_l1_flux, energy_ratio, peak_ratio)."""
import json, math, os, sys
import numpy as np

H = os.path.expanduser('~')
OUT = os.path.join(H, 'wl95_calc')
Z = np.load(os.path.join(H, 'rest_x92/results/nmf.npz'))
Mq = Z['Mq'].astype(np.float64)          # [6, bi, bo, c]
Cn, An = Z['Cn'], Z['An']                # [3 mat, 3 c, 64, 16]
SEL = json.load(open(os.path.join(H, 'rest_x94/results/select94.json')))['splits']
R86 = os.path.join(H, 'rest_ecro86')
NB = 64
PANEL = np.arange(1088, 1130)
zb = ((np.arange(NB) // 16) + 0.5) / 4                 # true layout NZ=4 x NPHI=16 (scene86; used by s94.operator and diag92.flux_w)
W94 = ((np.arange(NB) // 8) + 0.5) / 8 * (2 * np.pi / NB)   # an94.W as coded (NPHI=8, NZ=8) -> used by s94 field metrics
WT = zb * (2 * np.pi / NB)                              # physically consistent cos dOmega for the true layout
RANKS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
ZZ = zb[:, None] * zb[None, :]


def op_metric(A, M):                                    # s94.operator: pooled over (in, out, c)
    return float((np.abs(A - M) * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum())


def fro_c(A, M):                                        # diag92.mat_metrics 'fro' per channel (flux X)
    return [float(np.linalg.norm((A[..., c] - M[..., c]) * ZZ) / np.linalg.norm(M[..., c] * ZZ)) for c in range(3)]


def peak_op(A, M):                                      # diag92 'peak' (operator-level, per-row true peak), pooled channels
    X, Xt = A * ZZ[..., None], M * ZZ[..., None]
    pk = Xt.argmax(1)
    ii = np.arange(NB)
    return float(sum(X[ii, pk[:, c], c].sum() for c in range(3)) / sum(Xt[ii, pk[:, c], c].sum() for c in range(3)))


def svd_k(M, k):
    A = np.zeros_like(M)
    for c in range(3):
        U, s, Vt = np.linalg.svd(M[..., c] * ZZ)
        A[..., c] = ((U[:, :k] * s[:k]) @ Vt[:k]) / ZZ
    return A


def nmf_l1(M, Cm, Am, outer=40, inner=100):
    """IRLS weighted-Frobenius NMF towards min L1 of the flux matrix, initialised from WL92 NMF16 (feasible nonneg rank-16)."""
    A = np.zeros_like(M)
    for c in range(3):
        V = M[..., c] * ZZ
        Wf = Cm[c] * zb[:, None]; Hf = (Am[c] * zb[:, None]).T
        d = 1e-3 * V.mean(); eps = 1e-30
        for _ in range(outer):
            Om = 1.0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(inner):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + eps)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + eps)
        A[..., c] = (Wf @ Hf) / ZZ
    return A


def load_fields(split, pmat_id):
    rows = []
    for rec in SEL[split]['files']:
        z = np.load(os.path.join(R86, 'data', split, rec['file']))
        s = z['s'].astype(np.float64)                   # [L, N, 64, 3]
        sel = PANEL if pmat_id == 1 else np.nonzero(z['pmat'] == pmat_id)[0]
        if pmat_id == 1:
            assert (z['pmat'][PANEL] == 1).all()
        rows.append(s[:, sel].reshape(-1, NB, 3))       # rows (l, n)
    return rows                                         # list per item


def field_metrics(rows, A, M, W):
    acc = dict(l1=0., m=0., l=0., pl=0., pm=0.)
    for S in rows:
        Bl = np.einsum('kmc,mdc->kdc', S, A, optimize=True); Bm = np.einsum('kmc,mdc->kdc', S, M, optimize=True)
        acc['l1'] += float((np.abs(Bl - Bm) * W[None, :, None]).sum()); acc['m'] += float((Bm * W[None, :, None]).sum())
        acc['l'] += float((Bl * W[None, :, None]).sum())
        pk = Bm.argmax(1)[:, None, :]
        acc['pl'] += float(np.take_along_axis(Bl, pk, 1).sum()); acc['pm'] += float(np.take_along_axis(Bm, pk, 1).sum())
    return {'field_rel_l1': acc['l1'] / acc['m'], 'energy': acc['l'] / acc['m'], 'peak': acc['pl'] / acc['pm']}


def rrr(rows, M, W, k, rtol=1e-10):
    """Field-adapted rank-k operator: argmin_rank(A)<=k sum ||S (A - M) diag(W)||_F^2 over the pooled training fields, per channel.
    S = U s V^T; Z = best rank-k of diag(s) V^T M D; A = V diag(1/s) Z D^-1 (zero on the out-of-span incident subspace)."""
    A = np.zeros_like(M); info = []
    D = W
    for c in range(3):
        S = np.concatenate([r[..., c] for r in rows], 0)
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        keep = s > rtol * s[0]
        s, Vt = s[keep], Vt[keep]
        P = s[:, None] * Vt
        Y = P @ M[..., c] * D[None, :]
        u, sg, vt = np.linalg.svd(Y, full_matrices=False)
        Zk = (u[:, :k] * sg[:k]) @ vt[:k]
        A[..., c] = (Vt.T @ (Zk / s[:, None])) / D[None, :]
        e = s ** 2 / (s ** 2).sum()
        info.append({'span_rank_tol': int(keep.sum()), 'rank99': int(np.searchsorted(np.cumsum(e), 0.99) + 1),
                     'rank999': int(np.searchsorted(np.cumsum(e), 0.999) + 1)})
    return A, info


def run():
    os.makedirs(OUT, exist_ok=True)
    res = {'note': 'W94 = an94.W as coded (NPHI 8, NZ 8); WT = true-layout cos dOmega; zb = true ring cosine', 'materials': {}}
    for m, name in ((1, 'S1_panel'), (0, 'B_floor'), (2, 'S2_sphere')):
        M = Mq[m]
        tr, va = load_fields('train', m), load_fields('val', m)
        o = {'n_rows_train': int(sum(r.shape[0] for r in tr)), 'n_rows_val': int(sum(r.shape[0] for r in va)),
             'albedo_bin_max_W94': float((M * W94[None, :, None]).sum(1).max()), 'albedo_bin_max_WT': float((M * WT[None, :, None]).sum(1).max())}
        arms = {f'svd{k}': svd_k(M, k) for k in RANKS}
        arms['svd16_clip0'] = np.maximum(arms['svd16'], 0)
        arms['nmf16_wl92'] = np.stack([Cn[m, c] @ An[m, c].T for c in range(3)], -1)
        arms['nmf16_l1irls'] = nmf_l1(M, Cn[m], An[m])
        arms['zero'] = np.zeros_like(M)
        for W, wn in ((W94, 'W94'), (WT, 'WT')):
            for k in (4, 8, 16, 24, 32):
                A, info = rrr(tr, M, W, k)
                arms[f'rrr{k}_{wn}'] = A
                o[f'rrr_info_{wn}'] = info
        A16, _ = rrr(tr, M, W94, 16)
        arms['rrr16_W94_clip0'] = np.maximum(A16, 0)
        # per-panel-patch-index field-adapted rank-16 (position-conditioned; S1 only)
        tab = {}
        for an, A in arms.items():
            r = {'op_l1': op_metric(A, M), 'fro_c': fro_c(A, M), 'op_peak': peak_op(A, M),
                 'mass': float((A * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum()), 'min_entry_rel': float(A.min() / M.max())}
            for W, wn in ((W94, 'W94'), (WT, 'WT')):
                r[f'train_{wn}'] = field_metrics(tr, A, M, W); r[f'val_{wn}'] = field_metrics(va, A, M, W)
            tab[an] = r
        if m == 1:
            pp = {'val_W94': dict(l1=0., m=0., l=0., pl=0., pm=0.)}
            Ap = []
            for j in range(len(PANEL)):
                trj = [r.reshape(-1, len(PANEL), NB, 3)[:, j] for r in tr]
                vaj = [r.reshape(-1, len(PANEL), NB, 3)[:, j] for r in va]
                A, _ = rrr(trj, M, W94, 16)
                fm = field_metrics(vaj, A, M, W94)
                Ap.append((A, vaj))
            # pooled val metric across patch-specific operators
            acc = dict(l1=0., m=0., l=0., pl=0., pm=0.)
            for A, vaj in Ap:
                for S in vaj:
                    Bl = np.einsum('kmc,mdc->kdc', S, A, optimize=True); Bm = np.einsum('kmc,mdc->kdc', S, M, optimize=True)
                    acc['l1'] += float((np.abs(Bl - Bm) * W94[None, :, None]).sum()); acc['m'] += float((Bm * W94[None, :, None]).sum())
                    acc['l'] += float((Bl * W94[None, :, None]).sum())
                    pk = Bm.argmax(1)[:, None, :]
                    acc['pl'] += float(np.take_along_axis(Bl, pk, 1).sum()); acc['pm'] += float(np.take_along_axis(Bm, pk, 1).sum())
            o['rrr16_per_patch_W94_val'] = {'field_rel_l1': acc['l1'] / acc['m'], 'energy': acc['l'] / acc['m'], 'peak': acc['pl'] / acc['pm'],
                                           'op_l1_mean': float(np.mean([op_metric(A, M) for A, _ in Ap]))}
            # incident distribution diagnostics on the panel: ring shares of irradiance, number of lit bins per row
            S = np.concatenate(tr, 0)
            ring = np.array([S[:, 16 * i:16 * i + 16].sum() for i in range(4)]); o['train_ring_share'] = (ring / ring.sum()).tolist()
            o['train_bins_lit_per_row_median'] = float(np.median((S.sum(2) > 0).sum(1)))
            o['train_top1_bin_share_median'] = float(np.median(S.sum(2).max(1) / np.maximum(S.sum(2).sum(1), 1e-30)))
            np.savez_compressed(os.path.join(OUT, 'S1_ops.npz'), **{k: v for k, v in arms.items()})
        o['arms'] = tab
        res['materials'][name] = o
        print('done', name, flush=True)
        json.dump(res, open(os.path.join(OUT, 'wl95_calc.json'), 'w'), indent=1)
    json.dump(res, open(os.path.join(OUT, 'wl95_calc.json'), 'w'), indent=1)


if __name__ == '__main__':
    run()
