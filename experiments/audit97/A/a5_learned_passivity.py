"""WL97 audit, Track A, check A5 (server, torch + Dr.Jit, read-only inputs): where is the learned ECRO local operator non-passive?

WL95A reports only the MAX over (42 S1 patches x 64 incident bins x 3 channels) of the directional albedo
  alb[n, bin_in, c] = sum_bo (sum_r c[n, bin_in, c, r] a[n, bo, c, r]) ZB[bo] DOMEGA
of the frozen WL92A OWN models (2.26 / 3.08). This script recomputes that number with the production functions (a95.setup, a95.learned
semantics via mat94.load_frozen + model86.factors) and adds the distribution: by incident ring, on / off the incident bins that carry
CK2 irradiance, on / off the bins excited in training (WL94D train_bins.npy), irradiance-weighted (the energy the CK2 light actually
sees), and the same for every material group (Lambert patches have a known answer: rho).
Nothing is written outside the given output directory.
Run:  $PY a5_learned_passivity.py <R_own_scratch> <R92A> <train_bins.npy> <out.json>
"""
import json, os, sys
import numpy as np

CODE = os.path.expanduser('~/rest_x96a/code/experiments')
sys.path.insert(0, os.path.join(CODE, 'ecro_attr95a'))
import a95 as A95  # noqa: E402

R, R92A, TB, OUT = sys.argv[1:5]
PT, S, A, FX, ctx, lt = A95.setup(R)
import torch  # noqa: E402
import ecro86 as EC  # noqa: E402
import model86 as M86  # noqa: E402
MT = A95.MT
G = ctx.G
st = EC.build_state(ctx)
s16 = ctx.source(lt).astype(np.float64)                       # [N, NB, 3]
WT = A95.WT; NB = 64
ring = np.arange(NB) // 16
tb = np.load(TB).astype(bool)                                 # [NB_in, 3] incident bins of the panel excited in the WL94D training items
Mq = MT.mq_ext().astype(np.float64)                           # [mat, in, out, 3]
res = {'inputs': {'R92A': R92A, 'train_bins': TB, 'N': int(G.N)}, 'M64': {}, 'learned': {}}
for m, nm in ((0, 'floor_B'), (1, 'panel_S1'), (2, 'sphere_S2')):
    alb = (Mq[m] * WT[None, :, None]).sum(1)
    res['M64'][nm] = {'albedo_max': float(alb.max()), 'albedo_mean': float(alb.mean()), 'albedo_by_ring_mean': [float(alb[ring == r].mean()) for r in range(4)]}
groups = {'panel_S1': np.nonzero(G.PMAT == 1)[0], 'floor_B': np.nonzero(G.PMAT == 0)[0], 'sphere_S2': np.nonzero(G.PMAT == 2)[0],
          'lambert_0.7': np.nonzero(G.PMAT == 3)[0], 'boxA_0.5': np.nonzero(G.PMAT == 4)[0], 'boxB_0.6': np.nonzero(G.PMAT == 5)[0]}
dev = torch.device('cuda')
for sd in (8601, 8602):
    model = MT.load_frozen(R92A, sd)
    gs = M86.GeoState(G, st.hit, st.dbin, dev)
    with torch.no_grad():
        F = model.factors(gs)
        c, a = F['c'].double().cpu().numpy(), F['a'].double().cpu().numpy()            # [N, NB, 3, R]
    o = {}
    for gn, idx in groups.items():
        lop = np.einsum('nmcr,ndcr->nmdc', c[idx], a[idx])                             # [n, in, out, 3]
        alb = (lop * WT[None, None, :, None]).sum(2)                                   # [n, in, 3]
        s = s16[idx]                                                                   # [n, in, 3]
        lit = s > 0
        d = {'n_patches': int(len(idx)), 'albedo_max': float(alb.max()), 'albedo_mean': float(alb.mean()), 'albedo_median': float(np.median(alb)),
             'albedo_p90': float(np.quantile(alb, 0.9)), 'albedo_p99': float(np.quantile(alb, 0.99)), 'frac_rows_gt1': float((alb > 1 + 1e-6).mean()),
             'frac_patches_any_gt1': float((alb.reshape(len(idx), -1).max(1) > 1 + 1e-6).mean()),
             'by_incident_ring': {str(r): {'mean': float(alb[:, ring == r].mean()), 'max': float(alb[:, ring == r].max()), 'frac_gt1': float((alb[:, ring == r] > 1 + 1e-6).mean())}
                                  for r in range(4)},
             'by_channel_max': [float(alb[..., ch].max()) for ch in range(3)],
             'argmax': [int(v) for v in np.unravel_index(alb.argmax(), alb.shape)],
             'ck2_lit_rows': int(lit.sum()), 'ck2_lit_frac_gt1': float((alb[lit] > 1 + 1e-6).mean()) if lit.any() else None,
             'ck2_lit_albedo_max': float(alb[lit].max()) if lit.any() else None,
             'ck2_irradiance_weighted_albedo_rgb': [float((alb[..., ch] * s[..., ch]).sum() / max(s[..., ch].sum(), 1e-30)) for ch in range(3)],
             'ck2_irradiance_share_by_ring': [float(s[:, ring == r].sum() / max(s.sum(), 1e-30)) for r in range(4)]}
        if gn == 'panel_S1':
            m64 = (Mq[1] * WT[None, :, None]).sum(1)                                   # [in, 3]
            d['ck2_irradiance_weighted_albedo_M64_rgb'] = [float((m64[None, :, ch] * s[..., ch]).sum() / max(s[..., ch].sum(), 1e-30)) for ch in range(3)]
            tbm = np.broadcast_to(tb[None], alb.shape)
            d['train_excited_rows_frac'] = float(tb.mean())
            d['frac_gt1_on_train_excited'] = float((alb[tbm] > 1 + 1e-6).mean()); d['frac_gt1_off_train_excited'] = float((alb[~tbm] > 1 + 1e-6).mean()) if (~tbm).any() else None
            d['albedo_max_on_train_excited'] = float(alb[tbm].max()); d['albedo_max_off_train_excited'] = float(alb[~tbm].max()) if (~tbm).any() else None
            d['albedo_mean_on_train_excited'] = float(alb[tbm].mean())
            # per incident bin (mean over the 42 patches, green) for the report
            d['albedo_by_in_bin_green_mean_over_patches'] = [float(v) for v in alb[..., 1].mean(0)]
            d['M64_albedo_by_in_bin_green'] = [float(v) for v in m64[:, 1]]
        o[gn] = d
    res['learned'][str(sd)] = o
    print(sd, json.dumps({g: {k: v for k, v in d.items() if k in ('albedo_max', 'albedo_mean', 'frac_rows_gt1', 'ck2_lit_frac_gt1', 'ck2_irradiance_weighted_albedo_rgb')} for g, d in o.items()}), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('DONE', flush=True)
os._exit(0)
