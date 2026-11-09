import os, sys, json, numpy as np
sys.path.insert(0, os.path.expanduser('~/rest_closure80d/code/experiments/fnto_closure80d'))
sys.argv = ['x', os.path.expanduser('~/rest_compiled80c/results/smoke'), '/tmp/x', 'diag']
import run80d as RD
import torch
E, C = RD.E, RD.C; dev = RD.dev_t
wf, uf, wc, uc = RD.fit_grids()
u2 = E.strat_u(128, 256, 80460, dev)
th = np.concatenate([np.linspace(0, 89, 90), np.linspace(89.02, 89.999, 50)])
wfine = E.wi_grid(th, np.linspace(0, 90, 91), dev)
out = {}
cand = np.load(os.path.expanduser('~/rest_closure80d/results/candidate.npz'))
for name, z in [('candidate', torch.from_numpy(cand['zC']).to(dev)), ('adversarial_candidate', RD.adversarial_raw(80430).to(dev))]:
    q, M, idx, am, se, dt = RD.compile_candidate(z, wc, uc)
    B = z.shape[0]; chunk = max(1, int(2.5e7 // (B * 7 * u2.shape[0])))
    a_all, s_all = [], []
    with torch.no_grad():
        for s in range(0, wfine.shape[0], chunk):
            A, sA = E.lobe_albedo(q['au'], q['av'], q['f0'], wfine[s:s+chunk], u2, with_se=True)
            a_all.append(E.mixture_albedo(q['lam'], A)); s_all.append(torch.einsum('bcr,brw->bcw', q['lam'][:, :, 1:], sA))
    a = torch.cat(a_all, -1); sa = torch.cat(s_all, -1)
    amax, imax = a.max(-1)
    ex = (a - 1 > 3 * sa) & (a > 1.0)
    T = np.repeat(th, 91); PH = np.tile(np.linspace(0, 90, 91), len(th))
    viol = []
    for b in range(B):
        for c in range(3):
            if ex[b, c].any():
                i = int(imax[b, c]); viol.append({'closure': b, 'channel': c, 'albedo_max': float(amax[b, c]), 'theta': float(T[i]), 'phi': float(PH[i]), 'se': float(sa[b, c, i]), 'M_compiled': float(M[b, c])})
    out[name] = {'max_albedo_fine_grid': float(amax.max()), 'n_closure_channels_exceeding_1_by_3se': len(viol),
                 'theta_of_max_global': float(T[int(imax.flatten()[int(amax.flatten().argmax())])]), 'violations': viol[:20]}
    print(name, json.dumps(out[name])[:1500], flush=True)
json.dump(out, open(os.path.expanduser('~/rest_closure80d/results/diag_passivity_finegrid.json'), 'w'), indent=1)
os._exit(0)
