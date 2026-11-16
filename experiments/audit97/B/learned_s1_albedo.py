"""WL97 audit track B (server, read-only inputs): where is the learned S1 (panel) local operator a c^T non-passive, and how much training
irradiance did those incident bins receive?

  python learned_s1_albedo.py <CODE/experiments> <R_own (my dir)> <R92A> <R86> <out_json>

- CK2 state (mat94.ctx_ck2, default materials) and N_TRAIN_STATES stored WL86 training states (hit / dbin from the npz).
- own_s8601 / own_s8602 (WL92A, loaded read-only through mat94.load_frozen).
- Directional albedo per (S1 patch, incident bin m, channel): sum_d A[m, d] W_T[d], A = sum_r c[m, r] a[d, r]  (the a95.op_metrics definition).
- Panel direct-irradiance share per incident bin over the WL92A training population (stored source encodings s, 928 files x 4 lights).
Nothing is trained; nothing is written outside <R_own> / <out_json>.
"""
import glob, json, math, os, sys
import numpy as np

CODE, R, R92A, R86, OUT = sys.argv[1:6]
sys.path.insert(0, os.path.join(CODE, 'ecro_mat94'))
import mat94 as MT  # noqa: E402
import torch  # noqa: E402
import model86 as M86  # noqa: E402

NB = 64
S1 = np.arange(1088, 1130)
ZB = ((np.arange(NB) // 16) + 0.5) / 4
WT = ZB * (2 * math.pi / NB)
N_TRAIN_STATES = 48
dev = torch.device('cuda')


def op_albedo(c, a):
    A = np.einsum('jmcr,jdcr->jmdc', c, a, optimize=True)          # [42, in, out, 3]
    return (A * WT[None, None, :, None]).sum(2)                    # [42, in, 3]


def summarise(alb, lit_bins=None):
    o = {'max': float(alb.max()), 'frac_entries_gt_1': float((alb > 1 + 1e-6).mean()),
         'argmax_patch_bin_channel': [int(v) for v in np.unravel_index(alb.argmax(), alb.shape)],
         'per_ring_max': [float(alb[:, 16 * r:16 * r + 16].max()) for r in range(4)],
         'per_ring_mean': [float(alb[:, 16 * r:16 * r + 16].mean()) for r in range(4)],
         'per_ring_frac_gt_1': [float((alb[:, 16 * r:16 * r + 16] > 1 + 1e-6).mean()) for r in range(4)],
         'per_bin_max': alb.max((0, 2)).tolist(), 'per_bin_mean': alb.mean((0, 2)).tolist()}
    if lit_bins is not None:
        o['lit_bins'] = [int(b) for b in lit_bins]
        o['max_on_lit_bins'] = float(alb[:, lit_bins].max())
        o['mean_on_lit_bins'] = float(alb[:, lit_bins].mean())
    return o


def main():
    out = {}
    ex = {e[0] for e in json.load(open(os.path.join(R92A, 'results', 'exclusion.json')))['excluded']['train']}
    fns = [f for f in sorted(glob.glob(os.path.join(R86, 'data', 'train', 'g*.npz'))) if os.path.basename(f) not in ex]
    assert len(fns) == 928
    # ---- panel irradiance per incident bin over the OWN training population (stored encodings)
    binE = np.zeros(NB); per_light_ring0 = []; n_l = 0
    for fn in fns:
        z = np.load(fn)
        s = z['s'][:, S1].astype(np.float64)                       # [4, 42, 64, 3]
        e = s.sum((1, 3))                                          # [4, 64]
        binE += e.sum(0)
        tot = e.sum(1)
        for l in range(4):
            n_l += 1
            if tot[l] > 0:
                per_light_ring0.append(float(e[l, :16].sum() / tot[l]))
    out['train_P92A'] = {'files': len(fns), 'lightings': n_l, 'panel_irradiance_share_per_incident_bin': (binE / binE.sum()).tolist(),
                         'ring_share': [float(binE[16 * r:16 * r + 16].sum() / binE.sum()) for r in range(4)],
                         'lightings_panel_lit': len(per_light_ring0),
                         'lightings_ring0_share_ge_0.9': int(sum(v >= 0.9 for v in per_light_ring0)),
                         'lightings_ring0_share_eq_1': int(sum(v >= 1 - 1e-9 for v in per_light_ring0))}
    print('TRAIN ring share', out['train_P92A']['ring_share'], flush=True)
    # ---- CK2 state
    PT, S, A, FX, ctx, lt = MT.ctx_ck2(R)
    MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
    hit, dbin, _ = ctx.state_rays()
    s_ck = ctx.source(lt)[S1].astype(np.float64)                   # [42, 64, 3]
    eck = s_ck.sum((0, 2))
    lit = np.nonzero(eck > 0)[0]
    out['CK2'] = {'panel_irradiance_share_per_incident_bin': (eck / eck.sum()).tolist(), 'lit_bins': [int(b) for b in lit],
                  'train_irradiance_share_of_CK2_lit_bins': float((binE / binE.sum())[lit].sum())}
    print('CK2 lit bins', lit.tolist(), 'train share of those bins', out['CK2']['train_irradiance_share_of_CK2_lit_bins'], flush=True)
    gs_ck = M86.GeoState(ctx.G, hit, dbin, dev)
    # ---- training states (stored)
    idx = np.linspace(0, len(fns) - 1, N_TRAIN_STATES).round().astype(int)
    gss = []
    for i in idx:
        z = np.load(fns[i])
        g = type('G', (), {})()
        g.N = z['pc'].shape[0]
        g.PC, g.PN, g.PT, g.PAREA, g.PMAT = z['pc'], z['pn'], z['pt'], z['parea'], z['pmat']
        gss.append(M86.GeoState(g, z['hit'], z['dbin'], dev))
    for seed in (8601, 8602):
        model = MT.load_frozen(R92A, seed)
        with torch.no_grad():
            F = model.factors(gs_ck)
            alb_ck = op_albedo(F['c'][S1].double().cpu().numpy(), F['a'][S1].double().cpu().numpy())
            albs = []
            for gs in gss:
                F = model.factors(gs)
                albs.append(op_albedo(F['c'][S1].double().cpu().numpy(), F['a'][S1].double().cpu().numpy()))
        albs = np.stack(albs)                                      # [T, 42, 64, 3]
        o = {'CK2': summarise(alb_ck, lit), 'train_states_pooled': summarise(albs.reshape(-1, NB, 3)),
             'train_states_max_per_state': albs.max((1, 2, 3)).tolist(),
             'train_states_frac_states_nonpassive': float((albs.max((1, 2, 3)) > 1 + 1e-6).mean())}
        # relation between training irradiance share of a bin and its albedo excess
        share = binE / binE.sum()
        pb = albs.max((0, 1, 3))
        o['bins_gt_1_train_states'] = [int(b) for b in np.nonzero(pb > 1 + 1e-6)[0]]
        o['train_irradiance_share_of_bins_gt_1'] = float(share[pb > 1 + 1e-6].sum())
        o['spearman_bin_share_vs_bin_max_albedo'] = float(np.corrcoef(np.argsort(np.argsort(share)), np.argsort(np.argsort(pb)))[0, 1])
        out[f'own_s{seed}'] = o
        print(seed, 'CK2 max', o['CK2']['max'], 'ring max', o['CK2']['per_ring_max'], 'lit-bin max', o['CK2']['max_on_lit_bins'],
              '| train-state pooled max', o['train_states_pooled']['max'], 'ring max', o['train_states_pooled']['per_ring_max'],
              'frac states nonpassive', o['train_states_frac_states_nonpassive'], flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
