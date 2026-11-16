"""WL97 audit, phase 2, track J1 (server; torch + Dr.Jit; read-only inputs; one short GPU job).

Question: the WL95A passivity statement (max directional albedo 2.26 / 3.08) was measured on the WL92A OWN checkpoints.
Does it also hold for the WL86 deployed checkpoints (ecro_s8601 / ecro_s8602) and the WL92A SUM controls?

Same convention as experiments/ecro_attr95a/a95.py:op_metrics (alb[n, m, c] = sum_d (sum_r c[n,m,c,r] a[n,d,c,r]) W_T[d]),
same CK2 'none' geometry / state as WL95A.  Lambert patches (rho 0.7 / 0.5 / 0.6) are the normalisation control.
Nothing is written outside the output path.
Run:  $PY j1_wl86_passivity.py <scratch_R> <out.json>
"""
import hashlib, json, os, sys
import numpy as np

CODE = os.path.expanduser('~/rest_x96a/code/experiments')
sys.path.insert(0, os.path.join(CODE, 'ecro_attr95a'))
import a95 as A95  # noqa: E402

R, OUT = sys.argv[1:3]
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
area = np.asarray(G.PAREA, np.float64).reshape(-1)
Mq = MT.mq_ext().astype(np.float64)
dev = torch.device('cuda')
gs = M86.GeoState(G, st.hit, st.dbin, dev)
groups = {'panel_S1': np.nonzero(G.PMAT == 1)[0], 'floor_B': np.nonzero(G.PMAT == 0)[0], 'sphere_S2': np.nonzero(G.PMAT == 2)[0],
          'lambert_0.7': np.nonzero(G.PMAT == 3)[0], 'boxA_0.5': np.nonzero(G.PMAT == 4)[0], 'boxB_0.6': np.nonzero(G.PMAT == 5)[0]}
CK = {'wl86_ecro_s8601': '~/rest_ecro86/results/models/ecro_s8601.pt', 'wl86_ecro_s8602': '~/rest_ecro86/results/models/ecro_s8602.pt',
      'wl92a_own_s8601': '~/rest_x92a/results/models/own_s8601.pt', 'wl92a_own_s8602': '~/rest_x92a/results/models/own_s8602.pt',
      'wl92a_sum_s8601': '~/rest_x92a/results/models/sum_s8601.pt', 'wl92a_sum_s8602': '~/rest_x92a/results/models/sum_s8602.pt'}
res = {'M64_S1_albedo_max': float((Mq[1] * WT[None, :, None]).sum(1).max()), 'ck': {}}
for nm, p in CK.items():
    p = os.path.expanduser(p)
    sha = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
    m = M86.ECRO().to(dev)
    m.load_state_dict(torch.load(p, map_location=dev))
    m = MT.extend_buffer(m).eval()
    with torch.no_grad():
        F = m.factors(gs)
        c, a = F['c'].double().cpu().numpy(), F['a'].double().cpu().numpy()
    o = {'sha256_16': sha}
    for gn, idx in groups.items():
        lop = np.einsum('nmcr,ndcr->nmdc', c[idx], a[idx])
        alb = (lop * WT[None, None, :, None]).sum(2)                                   # [n, in, 3]
        s = s16[idx]
        lit = s > 0
        ex = (np.einsum('nmc,nmdc->ndc', s, lop) * WT[None, :, None]).sum((1, 2))      # once-scattered exitance per patch (rgb summed)
        ir = s.sum((1, 2))
        o[gn] = {'albedo_max': float(alb.max()), 'albedo_mean': float(alb.mean()), 'albedo_p90': float(np.quantile(alb, 0.9)),
                 'frac_rows_gt1': float((alb > 1 + 1e-6).mean()), 'frac_patches_any_gt1': float((alb.reshape(len(idx), -1).max(1) > 1 + 1e-6).mean()),
                 'by_ring_max': [float(alb[:, ring == r].max()) for r in range(4)], 'by_ring_frac_gt1': [float((alb[:, ring == r] > 1 + 1e-6).mean()) for r in range(4)],
                 'ck2_lit_frac_gt1': float((alb[lit] > 1 + 1e-6).mean()) if lit.any() else None,
                 'ck2_lit_albedo_max': float(alb[lit].max()) if lit.any() else None,
                 'ck2_exitance_over_irradiance': float((ex * area[idx]).sum() / max((ir * area[idx]).sum(), 1e-30)),
                 'ck2_patches_out_gt_in': int((ex > ir * (1 + 1e-6)).sum()), 'n_patches': int(len(idx)), 'min_entry': float(lop.min())}
    res['ck'][nm] = o
    print(nm, sha, json.dumps({g: [round(d['albedo_max'], 4), round(d['albedo_mean'], 4), round(d['frac_rows_gt1'], 4)] for g, d in o.items() if g != 'sha256_16'}), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('DONE', flush=True)
os._exit(0)
