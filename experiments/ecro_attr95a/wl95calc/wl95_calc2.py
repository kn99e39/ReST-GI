"""WL95 add-on (CPU): field-adapted NONNEGATIVE rank-16 S1 operator targeting the field L1 metric.
Weighted-L1 IRLS NMF on M64 S1 with weights E_tr[bi] * W[bo] (E_tr = pooled training panel irradiance per incident bin), i.e. the
field-L1 objective under the measured near-single-bin incidence; init WL92 NMF16. Evaluated with the WL94D field metrics on train / val."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/wl95_calc'))
import wl95_calc as C

M = C.Mq[1]
tr, va = C.load_fields('train', 1), C.load_fields('val', 1)
out = {}
for W, wn in ((C.W94, 'W94'),):
    E = sum(r.sum(0) for r in tr)                      # [64 in, 3]
    Ev = sum(r.sum(0) for r in va)
    out['E_share_train_by_ring'] = [float(E[16 * i:16 * i + 16].sum() / E.sum()) for i in range(4)]
    out['E_share_val_by_ring'] = [float(Ev[16 * i:16 * i + 16].sum() / Ev.sum()) for i in range(4)]
    out['bins_holding_90pct_train_E'] = int(np.searchsorted(np.cumsum(np.sort(E.sum(1))[::-1]) / E.sum(), 0.9) + 1)
    # single-bin approximation check: sum_k sum_d |sum_m s_km X_md| vs sum_m E_m sum_d |X_md| for X = NMF16 - M
    A16 = np.stack([C.Cn[1, c] @ C.An[1, c].T for c in range(3)], -1)
    num_exact = sum(float((np.abs(np.einsum('kmc,mdc->kdc', r, A16 - M, optimize=True)) * W[None, :, None]).sum()) for r in tr)
    num_sb = float((E[:, None, :] * np.abs(A16 - M) * W[None, :, None]).sum())
    out['single_bin_approx_ratio_nmf16'] = num_exact / num_sb
    A = np.zeros_like(M)
    for c in range(3):
        V = M[..., c]
        Om0 = E[:, c][:, None] * W[None, :]
        Wf = C.Cn[1, c].copy(); Hf = C.An[1, c].T.copy()
        d = 1e-3 * V.mean(); eps = 1e-30
        for _ in range(60):
            Om = Om0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(100):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + eps)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + eps)
        A[..., c] = Wf @ Hf
    out['fa_nmf16_l1'] = {'op_l1': C.op_metric(A, M), 'fro_c': C.fro_c(A, M), 'op_peak': C.peak_op(A, M),
                          'train_W94': C.field_metrics(tr, A, M, C.W94), 'val_W94': C.field_metrics(va, A, M, C.W94),
                          'val_WT': C.field_metrics(va, A, M, C.WT)}
    # rank ladder of the same field-adapted L1 objective (init: truncated nonneg SVD-free random-start NMF via WL92-style MU)
    lad = {}
    for k in (4, 8, 16, 24, 32):
        Ak = np.zeros_like(M)
        for c in range(3):
            V = M[..., c]; Om0 = E[:, c][:, None] * W[None, :]
            rng = np.random.default_rng(9500 + k * 10 + c)
            sc = np.sqrt(V.mean() / k)
            Wf = rng.random((64, k)) * sc + 1e-12; Hf = rng.random((k, 64)) * sc + 1e-12
            for _ in range(300):                       # weighted-Frobenius warm start
                Hf *= (Wf.T @ (Om0 * V)) / (Wf.T @ (Om0 * (Wf @ Hf)) + 1e-30)
                Wf *= ((Om0 * V) @ Hf.T) / ((Om0 * (Wf @ Hf)) @ Hf.T + 1e-30)
            d = 1e-3 * V.mean()
            for _ in range(60):
                Om = Om0 / np.maximum(np.abs(V - Wf @ Hf), d)
                for _ in range(100):
                    Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + 1e-30)
                    Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + 1e-30)
            Ak[..., c] = Wf @ Hf
        lad[k] = {'op_l1': C.op_metric(Ak, M), 'op_peak': C.peak_op(Ak, M),
                  'train_W94': C.field_metrics(tr, Ak, M, C.W94), 'val_W94': C.field_metrics(va, Ak, M, C.W94)}
        print('rank', k, lad[k], flush=True)
    out['fa_nmf_l1_ladder_randinit'] = lad
json.dump(out, open(os.path.expanduser('~/wl95_calc/wl95_calc2.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'fa_nmf_l1_ladder_randinit'}, indent=1, default=float))
