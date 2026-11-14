"""WL95 add-on 2 (CPU): position-conditioned (per panel patch index) field-adapted NONNEGATIVE rank-16 S1 operator, weighted-L1 IRLS NMF
with weights E_n[bi] * W94[bo] (patch n's pooled training irradiance per incident bin), init WL92 NMF16; WL94D field metrics on val."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/wl95_calc'))
import wl95_calc as C
M = C.Mq[1]; NP = len(C.PANEL)
tr, va = C.load_fields('train', 1), C.load_fields('val', 1)
trp = [r.reshape(-1, NP, 64, 3) for r in tr]; vap = [r.reshape(-1, NP, 64, 3) for r in va]
acc = {s: dict(l1=0., m=0., l=0., pl=0., pm=0.) for s in ('train', 'val')}
opl = []
for j in range(NP):
    E = sum(r[:, j].sum(0) for r in trp)
    A = np.zeros_like(M)
    for c in range(3):
        V = M[..., c]; Om0 = E[:, c][:, None] * C.W94[None, :] + 1e-6 * E[:, c].max() * C.W94[None, :]
        Wf = C.Cn[1, c].copy(); Hf = C.An[1, c].T.copy(); d = 1e-3 * V.mean()
        for _ in range(30):
            Om = Om0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(100):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + 1e-30)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + 1e-30)
        A[..., c] = Wf @ Hf
    opl.append(C.op_metric(A, M))
    for s, rows in (('train', trp), ('val', vap)):
        for r in rows:
            S = r[:, j]
            Bl = np.einsum('kmc,mdc->kdc', S, A, optimize=True); Bm = np.einsum('kmc,mdc->kdc', S, M, optimize=True)
            a = acc[s]; Wd = C.W94[None, :, None]
            a['l1'] += float((np.abs(Bl - Bm) * Wd).sum()); a['m'] += float((Bm * Wd).sum()); a['l'] += float((Bl * Wd).sum())
            pk = Bm.argmax(1)[:, None, :]
            a['pl'] += float(np.take_along_axis(Bl, pk, 1).sum()); a['pm'] += float(np.take_along_axis(Bm, pk, 1).sum())
out = {s: {'field_rel_l1': a['l1'] / a['m'], 'energy': a['l'] / a['m'], 'peak': a['pl'] / a['pm']} for s, a in acc.items()}
out['op_l1_mean'] = float(np.mean(opl)); out['op_l1_range'] = [float(min(opl)), float(max(opl))]
json.dump(out, open(os.path.expanduser('~/wl95_calc/wl95_calc3.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
